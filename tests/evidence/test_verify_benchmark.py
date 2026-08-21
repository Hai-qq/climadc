from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from climadc.benchmark import BenchmarkRunner
from climadc.cli.scaffold import scaffold_study
from climadc.config import StudyConfig
from climadc.evidence.verify import verify_run
from climadc.reporting import ArtifactWriter


@pytest.fixture(scope="module")
def benchmark_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("verified-benchmark")
    config_path = scaffold_study(root / "study")
    result = BenchmarkRunner().run(StudyConfig.from_yaml(config_path))
    return ArtifactWriter().write(result, root / "runs")


def test_verify_run_accepts_benchmark_v2(benchmark_run: Path) -> None:
    report = verify_run(benchmark_run)
    assert report.valid, report.to_json()
    assert report.run_type == "benchmark"
    assert report.artifact_schema_version == "2"


def test_benchmark_v2_checksum_tamper_fails(benchmark_run: Path, tmp_path: Path) -> None:
    run = tmp_path / benchmark_run.name
    shutil.copytree(benchmark_run, run)
    path = run / "metrics.json"
    path.write_bytes(path.read_bytes() + b"\n")

    assert not verify_run(run).valid


def test_legacy_benchmark_v1_is_partial_but_readable(benchmark_run: Path, tmp_path: Path) -> None:
    run = tmp_path / benchmark_run.name
    shutil.copytree(benchmark_run, run)
    for name in ("run-manifest.json", "environment.json", "checksums.sha256"):
        (run / name).unlink()

    report = verify_run(run)
    assert report.valid
    assert report.legacy
    assert report.artifact_schema_version == "1"
    assert report.limitations
