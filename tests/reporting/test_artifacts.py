from __future__ import annotations

import os
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

import climadc.reporting.artifacts as artifacts_module
from climadc.benchmark import BenchmarkRunner
from climadc.cli.scaffold import scaffold_study
from climadc.config import StudyConfig
from climadc.errors import ConfigurationError
from climadc.reporting import ArtifactWriter, resolve_run_path, update_latest_pointer


REQUIRED = {
    "run.yaml",
    "lineage.json",
    "splits.parquet",
    "predictions.parquet",
    "metrics.json",
    "leakage-report.json",
    "dataset-card.md",
    "report.html",
}


def _result(tmp_path: Path):
    config_path = scaffold_study(tmp_path / "study")
    return BenchmarkRunner().run(StudyConfig.from_yaml(config_path))


def test_artifact_writer_emits_exact_nonempty_set_and_relative_latest(tmp_path: Path) -> None:
    result = replace(_result(tmp_path), started_at=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc))
    runs = tmp_path / "runs"

    run_path = ArtifactWriter().write(result, runs)

    assert {item.name for item in run_path.iterdir()} == REQUIRED
    assert all(item.stat().st_size > 0 for item in run_path.iterdir())
    latest = runs / "latest"
    assert latest.is_symlink()
    assert os.readlink(latest) == run_path.name
    assert resolve_run_path(latest) == run_path.resolve()

    run_manifest = yaml.safe_load((run_path / "run.yaml").read_text(encoding="utf-8"))
    lineage = json.loads((run_path / "lineage.json").read_text(encoding="utf-8"))
    for payload in (run_manifest, lineage):
        assert payload["climadc_version"] == "0.1.0a1"
        assert payload["input_hashes"] == result.input_hashes
        assert payload["config"] == result.config_snapshot
        assert payload["started_at"] == result.started_at.isoformat()


def test_pointer_helpers_cover_posix_and_windows_without_platform_mutation(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "run-1"
    run.mkdir(parents=True)

    update_latest_pointer(runs, run, windows=False)
    assert (runs / "latest").is_symlink()
    assert resolve_run_path(runs / "latest", windows=False) == run.resolve()

    (runs / "latest").unlink()
    update_latest_pointer(runs, run, windows=True)
    assert not (runs / "latest").is_symlink()
    assert (runs / "latest").read_text(encoding="utf-8") == "run-1\n"
    assert resolve_run_path(runs / "latest", windows=True) == run.resolve()


def test_artifact_failure_leaves_no_partial_run_or_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _result(tmp_path)
    runs = tmp_path / "runs"
    writer = ArtifactWriter()

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write_payloads", fail)
    with pytest.raises(ConfigurationError, match="Unable to write run artifacts"):
        writer.write(result, runs)

    assert not (runs / "latest").exists()
    assert list(runs.iterdir()) == []


def test_latest_failure_rolls_back_already_renamed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _result(tmp_path)
    runs = tmp_path / "runs"

    def fail(*args: object, **kwargs: object) -> None:
        raise ConfigurationError("pointer failure")

    monkeypatch.setattr(artifacts_module, "update_latest_pointer", fail)
    with pytest.raises(ConfigurationError, match="pointer failure"):
        ArtifactWriter().write(result, runs)

    assert not (runs / "latest").exists()
    assert list(runs.iterdir()) == []
