from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from climadc.cli.app import app
from climadc.errors import ConfigurationError
from climadc.reference import packaged_study_path, packaged_suite_path
from climadc.reporting import resolve_run_path
from climadc.replay import (
    ReplaySuiteArtifactWriter,
    ReplaySuiteConfig,
    ReplaySuiteResult,
    ReplaySuiteRunner,
)
from climadc.replay.artifacts import REPLAY_ARTIFACTS
from climadc.replay.suite_artifacts import REPLAY_SUITE_ARTIFACTS

STARTED = pd.Timestamp("2026-08-07T04:00:00Z")


@pytest.fixture(scope="module")
def suite_result() -> ReplaySuiteResult:
    config = ReplaySuiteConfig.from_yaml(packaged_suite_path())
    return ReplaySuiteRunner(clock=lambda: STARTED).run(config)


def _absolute_study_payload() -> dict[str, object]:
    fixture = packaged_study_path().parent
    payload = yaml.safe_load(packaged_study_path().read_text(encoding="utf-8"))
    for item in payload["inputs"].values():
        item["path"] = str(fixture / item["path"])
    payload["source_manifest"] = str(fixture / payload["source_manifest"])
    return payload


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def test_suite_config_resolves_distinct_safe_scenarios() -> None:
    config = ReplaySuiteConfig.from_yaml(packaged_suite_path())

    assert config.suite_id == "gb-london-policy-sensitivity"
    assert len(config.scenarios) == 4
    assert len({scenario.study for scenario in config.scenarios}) == 4
    assert all(scenario.study.is_absolute() for scenario in config.scenarios)


@pytest.mark.parametrize(
    ("first_id", "second_id", "expected"),
    [
        ("../escape", "valid", "scenario_id"),
        ("same", "same", "scenario_id values must be unique"),
    ],
)
def test_suite_config_rejects_unsafe_or_duplicate_ids(
    tmp_path: Path,
    first_id: str,
    second_id: str,
    expected: str,
) -> None:
    suite_path = tmp_path / "suite.yaml"
    _write_yaml(
        suite_path,
        {
            "schema_version": "1",
            "suite_id": "invalid-suite",
            "scenarios": [
                {
                    "scenario_id": first_id,
                    "description": "first",
                    "study": str(packaged_study_path()),
                },
                {
                    "scenario_id": second_id,
                    "description": "second",
                    "study": str(packaged_study_path().parent / "study-cost-dominant.yaml"),
                },
            ],
        },
    )

    with pytest.raises(ConfigurationError, match=expected):
        ReplaySuiteConfig.from_yaml(suite_path)


def test_suite_runner_reconstructs_equal_weight_robustness_and_pareto(
    suite_result: ReplaySuiteResult,
) -> None:
    scenario_metrics = suite_result.scenario_metrics
    robustness = suite_result.robustness_metrics.set_index("policy")

    assert suite_result.mode == "single_window"
    assert suite_result.currency == "GBP"
    assert len(suite_result.scenarios) == 4
    assert len(scenario_metrics) == 4 * 6
    assert scenario_metrics["feasible"].all()
    assert set(robustness.index) == set(suite_result.policies)
    assert (robustness["feasible_fraction"] == 1.0).all()
    assert robustness.loc["asap", "cost_improvement_fraction_of_feasible"] == 0.0
    assert robustness.loc["asap", "emissions_improvement_fraction_of_feasible"] == 0.0
    assert robustness.loc["asap", "peak_improvement_fraction_of_feasible"] == 0.0
    assert (
        scenario_metrics.loc[
            scenario_metrics["policy"] == "joint", "energy_cost_change_vs_asap"
        ].nunique()
        > 1
    )

    for policy in suite_result.policies:
        rows = scenario_metrics.loc[scenario_metrics["policy"] == policy]
        aggregate = robustness.loc[policy]
        assert aggregate["mean_energy_cost_change_vs_asap"] == pytest.approx(
            rows["energy_cost_change_vs_asap"].mean()
        )
        assert aggregate["worst_emissions_change_vs_asap_kgco2e"] == pytest.approx(
            rows["emissions_change_vs_asap_kgco2e"].max()
        )
        assert aggregate["worst_peak_change_vs_asap_kw"] == pytest.approx(
            rows["peak_change_vs_asap_kw"].max()
        )

    flagged = tuple(
        suite_result.robustness_metrics.loc[
            suite_result.robustness_metrics["pareto_efficient"], "policy"
        ]
    )
    assert suite_result.pareto_frontier == flagged
    assert suite_result.pareto_frontier
    assert len(suite_result.config_sha256) == 64

    scenario_metrics.loc[:, "policy"] = "mutated"
    assert "mutated" not in set(suite_result.scenario_metrics["policy"])


def test_suite_excludes_infeasible_scenario_from_means_and_pareto(tmp_path: Path) -> None:
    infeasible_payload = _absolute_study_payload()
    infeasible_payload["replay"]["it_capacity_kw"] = 301
    infeasible_study = tmp_path / "infeasible.yaml"
    _write_yaml(infeasible_study, infeasible_payload)
    suite_path = tmp_path / "suite.yaml"
    _write_yaml(
        suite_path,
        {
            "schema_version": "1",
            "suite_id": "feasibility-gate",
            "scenarios": [
                {
                    "scenario_id": "base",
                    "description": "feasible base",
                    "study": str(packaged_study_path()),
                },
                {
                    "scenario_id": "capacity-stress",
                    "description": "insufficient flexible capacity",
                    "study": str(infeasible_study),
                },
            ],
        },
    )

    result = ReplaySuiteRunner(clock=lambda: STARTED).run(ReplaySuiteConfig.from_yaml(suite_path))
    robustness = result.robustness_metrics
    stressed = result.scenario_metrics.loc[
        result.scenario_metrics["scenario_id"] == "capacity-stress"
    ]

    assert not stressed["feasible"].any()
    assert stressed["energy_cost_change_vs_asap"].isna().all()
    assert (robustness["feasible_scenarios"] == 1).all()
    assert (robustness["feasible_fraction"] == 0.5).all()
    assert not robustness["pareto_efficient"].any()
    assert result.pareto_frontier == ()


def test_suite_rejects_incomparable_replay_grids(tmp_path: Path) -> None:
    incompatible_payload = _absolute_study_payload()
    incompatible_payload["replay"]["horizon"] = "12h"
    incompatible_study = tmp_path / "incompatible.yaml"
    _write_yaml(incompatible_study, incompatible_payload)
    suite_path = tmp_path / "suite.yaml"
    _write_yaml(
        suite_path,
        {
            "schema_version": "1",
            "suite_id": "incomparable",
            "scenarios": [
                {
                    "scenario_id": "base",
                    "description": "24-hour base",
                    "study": str(packaged_study_path()),
                },
                {
                    "scenario_id": "short",
                    "description": "12-hour variant",
                    "study": str(incompatible_study),
                },
            ],
        },
    )

    with pytest.raises(ConfigurationError, match="must share horizon"):
        ReplaySuiteRunner(clock=lambda: STARTED).run(ReplaySuiteConfig.from_yaml(suite_path))


def test_suite_artifacts_include_auditable_subruns(
    tmp_path: Path,
    suite_result: ReplaySuiteResult,
) -> None:
    run_path = ReplaySuiteArtifactWriter().write(suite_result, tmp_path / "suite-runs")

    assert {path.name for path in run_path.iterdir()} == REPLAY_SUITE_ARTIFACTS
    index = json.loads((run_path / "scenario-index.json").read_text(encoding="utf-8"))
    robustness = json.loads((run_path / "robustness-metrics.json").read_text(encoding="utf-8"))
    pareto = json.loads((run_path / "pareto-frontier.json").read_text(encoding="utf-8"))
    report = (run_path / "report.html").read_text(encoding="utf-8")

    assert len(index["scenarios"]) == 4
    assert len(robustness["records"]) == 6
    assert all(
        record["cost_improvement_fraction_of_feasible"] is not None
        for record in robustness["records"]
    )
    assert pareto["policies"] == list(suite_result.pareto_frontier)
    assert "EQUAL-WEIGHT ROBUSTNESS STUDY" in report
    assert "not production savings or guarantees" in report
    assert "<script" not in report.lower()
    assert "<link" not in report.lower()
    for scenario in index["scenarios"]:
        scenario_run = run_path / scenario["relative_run_path"]
        assert {path.name for path in scenario_run.iterdir()} == REPLAY_ARTIFACTS
        assert resolve_run_path(scenario_run.parent / "latest") == scenario_run
        assert f"{scenario['relative_run_path']}/report.html" in report
    assert resolve_run_path(tmp_path / "suite-runs" / "latest") == run_path


def test_cli_runs_packaged_and_generic_robustness_suites(tmp_path: Path) -> None:
    runner = CliRunner()
    demo_help = runner.invoke(app, ["demo", "robustness-suite", "--help"])
    assert demo_help.exit_code == 0, demo_help.output
    assert "four-scenario" in demo_help.output
    demo = runner.invoke(
        app,
        [
            "demo",
            "robustness-suite",
            "--output-dir",
            str(tmp_path / "demo-runs"),
        ],
    )
    assert demo.exit_code == 0, demo.output
    demo_path = Path(demo.stdout.strip())
    generic = runner.invoke(
        app,
        [
            "replay-suite",
            str(packaged_suite_path()),
            "--output-dir",
            str(tmp_path / "generic-runs"),
        ],
    )
    assert generic.exit_code == 0, generic.output
    generic_path = Path(generic.stdout.strip())

    demo_metrics = json.loads((demo_path / "robustness-metrics.json").read_text(encoding="utf-8"))
    generic_metrics = json.loads(
        (generic_path / "robustness-metrics.json").read_text(encoding="utf-8")
    )
    assert demo_metrics["records"] == generic_metrics["records"]
    report = runner.invoke(app, ["report", str(tmp_path / "generic-runs" / "latest")])
    assert report.exit_code == 0, report.output
    assert Path(report.stdout.strip()) == generic_path / "report.html"
