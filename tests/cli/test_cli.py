from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import climadc.cli.app as app_module
from climadc.cli.app import app


def test_cli_init_validate_benchmark_report(tmp_path: Path) -> None:
    runner = CliRunner()
    project = tmp_path / "study"

    initialized = runner.invoke(app, ["init", str(project)])
    assert initialized.exit_code == 0, initialized.output
    assert runner.invoke(app, ["validate", str(project / "study.yaml")]).exit_code == 0

    benchmark = runner.invoke(app, ["benchmark", str(project / "study.yaml")])
    assert benchmark.exit_code == 0, benchmark.output
    run_path = Path(benchmark.stdout.strip())
    assert run_path.is_dir()

    report = runner.invoke(app, ["report", str(project / "runs" / "latest")])
    assert report.exit_code == 0, report.output
    assert Path(report.stdout.strip()) == run_path / "report.html"
    assert (project / "runs" / "latest" / "report.html").exists()


def test_cli_init_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    runner = CliRunner()
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert runner.invoke(app, ["init", str(first)]).exit_code == 0
    assert runner.invoke(app, ["init", str(second)]).exit_code == 0

    managed = [
        "climate.csv",
        "telemetry.csv",
        "workload.csv",
        "climate-card.yaml",
        "telemetry-card.yaml",
        "workload-card.yaml",
        "study.yaml",
    ]
    for name in managed:
        assert (first / name).read_bytes() == (second / name).read_bytes()

    repeated = runner.invoke(app, ["init", str(first)])
    assert repeated.exit_code != 0
    assert "not empty" in repeated.stderr


def test_cli_translates_only_climadc_errors(tmp_path: Path) -> None:
    runner = CliRunner()
    missing = runner.invoke(app, ["validate", str(tmp_path / "missing.yaml")])
    assert missing.exit_code != 0
    assert "Invalid study config" in missing.stderr
    assert "Traceback" not in missing.stderr


def test_cli_preserves_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.startswith("climadc ")


def test_cli_does_not_translate_unexpected_bugs(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "study.yaml"
    config.write_text("study_id: unused\n", encoding="utf-8")

    def fail(path: Path):
        raise RuntimeError("unexpected bug")

    monkeypatch.setattr(app_module.StudyConfig, "from_yaml", fail)
    result = CliRunner().invoke(app, ["validate", str(config)])

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
