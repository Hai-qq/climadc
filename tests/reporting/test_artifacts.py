from __future__ import annotations

import os
import json
from dataclasses import replace
from datetime import datetime, timezone
from html.parser import HTMLParser
from math import isfinite
from pathlib import Path

import pytest
import pandas as pd
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


class _ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: set[str] = set()

    def handle_starttag(self, tag: str, attrs) -> None:
        self.tags.add(tag)


def _assert_finite_json(value: object) -> None:
    if isinstance(value, float):
        assert isfinite(value)
    elif isinstance(value, dict):
        for nested in value.values():
            _assert_finite_json(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_finite_json(nested)


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
        assert payload["run_id"] == run_path.name
        assert payload["study_id"] == result.study_id
        assert payload["climadc_version"] == "0.1.0a1"
        assert payload["input_hashes"] == result.input_hashes
        assert payload["config"] == result.config_snapshot
        assert payload["started_at"] == result.started_at.isoformat()


def test_all_artifacts_are_parseable_truthful_and_finite(tmp_path: Path) -> None:
    result = replace(_result(tmp_path), started_at=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc))
    run_path = ArtifactWriter().write(result, tmp_path / "runs")

    run_manifest = yaml.safe_load((run_path / "run.yaml").read_text(encoding="utf-8"))
    lineage = json.loads((run_path / "lineage.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    leakage = json.loads((run_path / "leakage-report.json").read_text(encoding="utf-8"))
    splits = pd.read_parquet(run_path / "splits.parquet")
    predictions = pd.read_parquet(run_path / "predictions.parquet")
    cards = (run_path / "dataset-card.md").read_text(encoding="utf-8")
    report = (run_path / "report.html").read_text(encoding="utf-8")
    parser = _ReportParser()
    parser.feed(report)
    parser.close()

    assert set(run_manifest) >= {
        "run_id",
        "study_id",
        "climadc_version",
        "started_at",
        "config_sha256",
        "config",
        "input_hashes",
    }
    assert set(lineage) >= {
        "run_id",
        "study_id",
        "climadc_version",
        "started_at",
        "config_sha256",
        "config",
        "input_hashes",
        "split_ids",
        "model_ids",
    }
    assert metrics == result.metrics
    assert leakage["accepted_rows"] + leakage["rejected_rows"] == 96
    assert not splits.empty
    assert not predictions.empty
    assert cards.startswith("# Dataset cards\n")
    assert cards.count("\n## ") == 3
    assert {"html", "body", "h1", "pre"}.issubset(parser.tags)
    for payload in (run_manifest, lineage, metrics, leakage):
        _assert_finite_json(payload)
    for name in (
        "run.yaml",
        "lineage.json",
        "metrics.json",
        "leakage-report.json",
        "dataset-card.md",
        "report.html",
    ):
        text = (run_path / name).read_text(encoding="utf-8")
        assert "example.invalid" not in text
        assert "placeholder" not in text.lower()
        assert "NaN" not in text


def test_artifact_validation_parses_content_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _result(tmp_path)
    writer = ArtifactWriter()
    original = writer._write_payloads

    def corrupt(directory: Path, value, run_id: str) -> None:
        original(directory, value, run_id)
        (directory / "metrics.json").write_text("{not-json}\n", encoding="utf-8")

    monkeypatch.setattr(writer, "_write_payloads", corrupt)
    with pytest.raises(ConfigurationError, match="Invalid artifact content"):
        writer.write(result, tmp_path / "runs")
    assert list((tmp_path / "runs").iterdir()) == []


def test_report_escapes_untrusted_study_text(tmp_path: Path) -> None:
    result = replace(_result(tmp_path), study_id='<script>alert("x")</script>')
    run_path = ArtifactWriter().write(result, tmp_path / "runs")
    report = (run_path / "report.html").read_text(encoding="utf-8")

    assert '<script>alert("x")</script>' not in report
    assert "&lt;script&gt;alert(&#34;x&#34;)&lt;/script&gt;" in report


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


@pytest.mark.parametrize("windows", [False, True])
def test_update_latest_rejects_non_child_targets(tmp_path: Path, windows: bool) -> None:
    runs = tmp_path / "runs"
    outside = tmp_path / "outside"
    nested = runs / "nested" / "run-1"
    outside.mkdir()
    nested.mkdir(parents=True)

    for target in (outside, nested, Path("../outside"), outside.resolve()):
        with pytest.raises(ConfigurationError, match="direct child"):
            update_latest_pointer(runs, target, windows=windows)
    assert not (runs / "latest").exists()


@pytest.mark.parametrize(
    ("windows", "target"),
    [
        (True, "../outside"),
        (True, "nested/run-1"),
        (True, "nested\\run-1"),
        (True, "/tmp/outside"),
        (False, "../outside"),
        (False, "nested/run-1"),
        (False, "nested\\run-1"),
        (False, "/tmp/outside"),
    ],
)
def test_resolve_latest_rejects_traversal_and_external_targets(
    tmp_path: Path, windows: bool, target: str
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    pointer = runs / "latest"
    if windows:
        pointer.write_text(f"{target}\n", encoding="utf-8")
    else:
        pointer.symlink_to(target, target_is_directory=True)

    with pytest.raises(ConfigurationError, match="direct child"):
        resolve_run_path(pointer, windows=windows)


def test_resolve_latest_uses_selected_pointer_format(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "run-1"
    run.mkdir(parents=True)
    pointer = runs / "latest"
    pointer.write_text("run-1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="POSIX symlink"):
        resolve_run_path(pointer, windows=False)

    pointer.unlink()
    pointer.symlink_to("run-1", target_is_directory=True)
    with pytest.raises(ConfigurationError, match="Windows text"):
        resolve_run_path(pointer, windows=True)


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
