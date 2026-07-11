from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from climadc.benchmark import BenchmarkRunner
from climadc.contracts.frames import ClimateForecastFrame
from climadc.errors import ConfigurationError
from climadc.adapters.weatherdc import (
    DownloadManifest,
    DownloadRecord,
    SourceItem,
    SourceManifest,
    WeatherDCAdapter,
)
from climadc.reporting.artifacts import REQUIRED_ARTIFACTS
from examples.weatherdc_kasetsart.run import (
    _materialize_small_study,
    _reference_predictions,
    _last_available_value,
    prepare_conversion,
    run_small,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "weatherdc_small"


def test_reference_persistence_uses_availability_not_only_event_time() -> None:
    origin = pd.Timestamp("2026-01-01 01:00:00+00:00")
    rows = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2026-01-01 00:00:00+00:00", "2026-01-01 01:00:00+00:00"]
            ),
            "available_at": pd.to_datetime(
                ["2026-01-01 00:00:00+00:00", "2026-01-01 02:00:00+00:00"]
            ),
            "value": [10.0, 999.0],
        }
    )

    assert _last_available_value(rows, origin) == 10.0


def test_weatherdc_ols_rejects_delayed_training_label(tmp_path: Path) -> None:
    config, climate, telemetry = _materialize_small_study(tmp_path / "runs")
    base = BenchmarkRunner().run(config)
    origin = base.predictions.to_pandas()["issue_time"].max()
    first_train_time = base.splits.loc[base.splits["partition"] == "train", "timestamp"].min()
    delayed = telemetry.copy(deep=True)
    row = (delayed["metric"] == "cooling_power") & (delayed["event_time"] == first_train_time)
    delayed.loc[row, "available_at"] = origin + pd.Timedelta("1h")
    delayed.loc[row, "value"] = 999_999.0

    with pytest.raises(ValueError, match="causal training alignment"):
        _reference_predictions(base, climate, delayed)


def test_weatherdc_small_is_causal_weather_aware_and_auditable(tmp_path: Path) -> None:
    result, run_dir = run_small(tmp_path / "runs")

    metrics = result.metrics["cooling_power"]
    assert metrics["weatherdc"]["mae"] < metrics["persistence"]["mae"]
    assert result.leakage_audit.rejected_rows == 0
    assert result.decision is not None
    assert result.decision.metrics["energy_conservation_error"] < 1e-8
    assert {path.name for path in run_dir.iterdir()} == REQUIRED_ARTIFACTS

    predictions = result.predictions.to_pandas()
    reference = predictions.loc[
        predictions["model_id"].isin({"weatherdc--split-000", "weatherdc-persistence--split-000"})
    ]
    assert set(reference["model_id"]) == {
        "weatherdc--split-000",
        "weatherdc-persistence--split-000",
    }
    actual = pd.read_csv(
        Path(__file__).parents[1] / "fixtures" / "weatherdc_small" / "expected.csv",
        parse_dates=["event_time"],
    ).set_index("event_time")["cooling_power"]
    for key, model_id in (
        ("weatherdc", "weatherdc--split-000"),
        ("persistence", "weatherdc-persistence--split-000"),
    ):
        rows = reference.loc[reference["model_id"] == model_id]
        errors = [
            abs(float(row.value) - float(actual.loc[row.valid_time]))
            for row in rows.itertuples(index=False)
        ]
        assert metrics[key]["mae"] == pytest.approx(sum(errors) / len(errors))


def test_weatherdc_artifacts_bind_external_model_configuration_and_code(tmp_path: Path) -> None:
    result, run_dir = run_small(tmp_path / "runs")
    extension = result.config_snapshot["extensions"]["weatherdc_reference"]

    assert extension["features"] == ["temp", "humid", "solar"]
    assert extension["algorithm"] == "ordinary_least_squares"
    assert extension["training_boundary"] == "available_at <= common_origin"
    assert (
        result.input_hashes["weatherdc_reference_implementation"]
        == hashlib.sha256(
            (Path(__file__).parents[2] / "examples" / "weatherdc_kasetsart" / "run.py").read_bytes()
        ).hexdigest()
    )
    normalized = json.dumps(
        result.config_snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert result.config_sha256 == hashlib.sha256(normalized.encode()).hexdigest()

    run_manifest = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    lineage = json.loads((run_dir / "lineage.json").read_text(encoding="utf-8"))
    expected_models = {
        "persistence--split-000",
        "seasonal--split-000",
        "calendar-linear--split-000",
        "weatherdc--split-000",
        "weatherdc-persistence--split-000",
    }
    assert run_manifest["config"]["extensions"] == result.config_snapshot["extensions"]
    assert run_manifest["input_hashes"] == result.input_hashes
    assert set(lineage["model_ids"]) == expected_models


def test_full_conversion_persists_auditable_manifest_without_workload_or_benchmark(
    tmp_path: Path,
) -> None:
    payload = b"synthetic upstream stand-in\n"
    source = SourceItem(
        name="source.csv",
        url="https://data.example.org/source.csv",
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )

    class FakeAdapter:
        def download(self, cache_dir: Path, manifest: SourceManifest) -> DownloadManifest:
            assert manifest.items == (source,)
            cache_dir.mkdir(parents=True)
            path = cache_dir / source.name
            path.write_bytes(payload)
            return DownloadManifest(
                records=(DownloadRecord(source.name, path, source.sha256, source.bytes),)
            )

        def load(self, cache_dir: Path):
            climate, telemetry = WeatherDCAdapter().load(FIXTURE)
            rows = climate.to_pandas()
            rows["issue_time"] = rows["valid_time"]
            rows["available_at"] = rows["valid_time"]
            rows["source"] = "weatherdc:hii-observation"
            return ClimateForecastFrame.from_pandas(rows), telemetry

    ticks = iter([10.0, 12.5])
    study_dir, elapsed = prepare_conversion(
        tmp_path / "cache",
        adapter=FakeAdapter(),
        manifest=SourceManifest(items=(source,)),
        clock=lambda: next(ticks),
    )

    manifest = yaml.safe_load((study_dir / "conversion-manifest.yaml").read_text(encoding="utf-8"))
    assert elapsed == 2.5
    assert manifest["mode"] == "conversion_only"
    assert manifest["elapsed_seconds"] == 2.5
    assert manifest["benchmark_produced"] is False
    assert manifest["workload_produced"] is False
    assert manifest["observation_time_semantics"] == "issue_time = available_at = valid_time"
    assert manifest["sources"] == [
        {
            "name": source.name,
            "url": source.url,
            "bytes": source.bytes,
            "sha256": source.sha256,
        }
    ]
    assert set(manifest["outputs"]) == {"climate.csv", "telemetry.csv"}
    for name, digest in manifest["outputs"].items():
        assert digest == hashlib.sha256((study_dir / name).read_bytes()).hexdigest()
    assert not (study_dir / "workload.csv").exists()


def test_full_conversion_rejects_forecast_rows_before_writing_observation_manifest(
    tmp_path: Path,
) -> None:
    class ForecastAdapter:
        def download(self, cache_dir: Path, manifest: SourceManifest) -> DownloadManifest:
            return DownloadManifest(records=())

        def load(self, cache_dir: Path):
            return WeatherDCAdapter().load(FIXTURE)

    with pytest.raises(ConfigurationError, match="observation-only climate"):
        prepare_conversion(
            tmp_path / "cache",
            adapter=ForecastAdapter(),
            manifest=SourceManifest(items=()),
        )

    assert not (tmp_path / "cache" / "study" / "conversion-manifest.yaml").exists()


def test_documented_small_command_works_without_pythonpath(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            "examples/weatherdc_kasetsart/run.py",
            "--small",
            "--output-dir",
            str(tmp_path / "runs"),
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    run_dir = Path(completed.stdout.strip())
    assert {path.name for path in run_dir.iterdir()} == REQUIRED_ARTIFACTS
