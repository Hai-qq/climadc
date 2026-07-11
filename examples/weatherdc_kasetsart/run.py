from __future__ import annotations

import argparse
import hashlib
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd
import yaml

from climadc.adapters.weatherdc import (
    WEATHERDC_SOURCE_MANIFEST,
    DownloadManifest,
    SourceManifest,
    WeatherDCAdapter,
)
from climadc.benchmark import BenchmarkRunner, RunResult
from climadc.config import InputConfig, StudyConfig
from climadc.contracts.frames import (
    PREDICTION_COLUMNS,
    ClimateForecastFrame,
    DCTelemetryFrame,
    PredictionFrame,
)
from climadc.errors import ConfigurationError
from climadc.evaluation import point_metrics
from climadc.reporting import ArtifactWriter

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "benchmarks" / "weatherdc.yaml"
SMALL_FIXTURE = ROOT / "tests" / "fixtures" / "weatherdc_small"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
        newline="\n",
    )


def _card(
    name: str,
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    return {
        "name": name,
        "site": {
            "site_id": "weatherdc-kasetsart",
            "latitude": 13.84705,
            "longitude": 100.571942,
            "timezone": "Asia/Bangkok",
        },
        "source": {
            "provider": "ClimaDC project",
            "url": "https://github.com/Hai-qq/climadc",
            "license": "CC0-1.0",
            "redistribution_constraints": None,
        },
        "sha256": _sha256(path),
        "schema_version": "1.0",
        "time_start": start.isoformat(),
        "time_end": end.isoformat(),
        "sampling_frequency": "1h",
        "known_missing": [],
        "spatial_mismatch": [
            "Schema mirrors the WeatherDC station/DC relationship, but all small-mode values "
            "are synthetic."
        ],
        "quality_limitations": [
            "Deterministic project-generated fixture; not representative of real operations."
        ],
    }


def _materialize_small_study(output_dir: Path) -> tuple[StudyConfig, pd.DataFrame, pd.DataFrame]:
    study_dir = output_dir / ".inputs"
    study_dir.mkdir(parents=True, exist_ok=True)
    climate, telemetry = WeatherDCAdapter().load(SMALL_FIXTURE)
    climate_rows = climate.to_pandas()
    telemetry_rows = telemetry.to_pandas()
    workload_rows = pd.read_csv(SMALL_FIXTURE / "workload.csv")

    climate_path = study_dir / "climate.csv"
    telemetry_path = study_dir / "telemetry.csv"
    workload_path = study_dir / "workload.csv"
    climate_rows.to_csv(climate_path, index=False, lineterminator="\n")
    telemetry_rows.to_csv(telemetry_path, index=False, lineterminator="\n")
    workload_rows.to_csv(workload_path, index=False, lineterminator="\n")

    cards = (
        (
            "Synthetic WeatherDC climate forecast",
            climate_path,
            study_dir / "climate-card.yaml",
            cast(pd.Timestamp, climate_rows["valid_time"].min()),
            cast(pd.Timestamp, climate_rows["valid_time"].max()),
        ),
        (
            "Synthetic WeatherDC telemetry",
            telemetry_path,
            study_dir / "telemetry-card.yaml",
            cast(pd.Timestamp, telemetry_rows["event_time"].min()),
            cast(pd.Timestamp, telemetry_rows["event_time"].max()),
        ),
        (
            "Synthetic WeatherDC workload",
            workload_path,
            study_dir / "workload-card.yaml",
            pd.Timestamp(workload_rows["event_time"].min()),
            pd.Timestamp(workload_rows["event_time"].max()),
        ),
    )
    for name, path, card_path, start, end in cards:
        _write_yaml(card_path, _card(name, path, start, end))

    template = StudyConfig.from_yaml(CONFIG)
    config = template.model_copy(
        update={
            "climate": InputConfig(
                path=climate_path,
                format="csv",
                timezone="UTC",
                card=study_dir / "climate-card.yaml",
            ),
            "telemetry": InputConfig(
                path=telemetry_path,
                format="csv",
                timezone="UTC",
                card=study_dir / "telemetry-card.yaml",
            ),
            "workload": InputConfig(
                path=workload_path,
                format="csv",
                timezone="UTC",
                card=study_dir / "workload-card.yaml",
            ),
            "output_dir": output_dir,
        }
    )
    return config, climate_rows, telemetry_rows


def _last_available_value(rows: pd.DataFrame, origin: pd.Timestamp) -> float:
    legal = rows.loc[(rows["event_time"] <= origin) & (rows["available_at"] <= origin)].sort_values(
        ["event_time", "available_at"], kind="mergesort"
    )
    if legal.empty:
        raise ValueError("Persistence requires one target available at the decision origin")
    return float(legal.iloc[-1]["value"])


def _complete_alignment(
    label: str,
    requested: pd.DatetimeIndex,
    features: pd.DataFrame,
    targets: pd.Series,
) -> pd.DatetimeIndex:
    missing_features = requested.difference(features.index)
    missing_targets = requested.difference(targets.index)
    if requested.empty or not missing_features.empty or not missing_targets.empty:
        raise ValueError(
            f"Incomplete {label} alignment: requested={len(requested)}, "
            f"missing_features={len(missing_features)}, missing_targets={len(missing_targets)}"
        )
    return requested


def _reference_predictions(
    result: RunResult,
    climate: pd.DataFrame,
    telemetry: pd.DataFrame,
) -> tuple[PredictionFrame, dict[str, dict[str, float]]]:
    split = result.splits.loc[result.splits["split_id"] == "split-000"]
    train_times = pd.DatetimeIndex(split.loc[split["partition"] == "train", "timestamp"])
    test_times = pd.DatetimeIndex(split.loc[split["partition"] == "test", "timestamp"])
    base_predictions = result.predictions.to_pandas()
    origin = cast(pd.Timestamp, base_predictions["issue_time"].max())

    features = cast(
        pd.DataFrame,
        climate.loc[
            (climate["available_at"] <= origin)
            & climate["variable"].isin(["temp", "humid", "solar"])
        ].pivot(index="valid_time", columns="variable", values="value"),
    )
    target_rows = cast(
        pd.DataFrame,
        telemetry.loc[
            (telemetry["metric"] == "cooling_power") & (telemetry["quality"] == "observed")
        ],
    )
    causal_targets = cast(
        pd.Series,
        target_rows.loc[target_rows["available_at"] <= origin].set_index("event_time")["value"],
    )
    actual_targets = cast(pd.Series, target_rows.set_index("event_time")["value"])
    train_index = _complete_alignment("causal training", train_times, features, causal_targets)
    test_index = _complete_alignment("post-hoc test", test_times, features, actual_targets)
    train_x = np.column_stack(
        [np.ones(len(train_index)), features.loc[train_index, ["temp", "humid", "solar"]]]
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        train_x, causal_targets.loc[train_index].to_numpy(dtype=float), rcond=None
    )
    test_x = np.column_stack(
        [np.ones(len(test_index)), features.loc[test_index, ["temp", "humid", "solar"]]]
    )
    weatherdc_values = test_x @ coefficients
    persistence_values = np.full(len(test_index), _last_available_value(target_rows, origin))

    rows: list[dict[str, object]] = []
    for model_id, values in (
        ("weatherdc--split-000", weatherdc_values),
        ("weatherdc-persistence--split-000", persistence_values),
    ):
        for valid_time, value in zip(test_index, values, strict=True):
            rows.append(
                {
                    "site_id": "weatherdc-kasetsart",
                    "issue_time": origin,
                    "valid_time": valid_time,
                    "target": "cooling_power",
                    "value": float(value),
                    "unit": "kW",
                    "model_id": model_id,
                    "quantile": float("nan"),
                }
            )
    predictions = PredictionFrame.from_pandas(pd.DataFrame(rows, columns=PREDICTION_COLUMNS))
    actual = actual_targets.loc[test_index].to_numpy(dtype=float)
    metrics = {
        "weatherdc": point_metrics(actual, weatherdc_values),
        "persistence": point_metrics(actual, persistence_values),
    }
    return predictions, metrics


def run_small(output_dir: Path) -> tuple[RunResult, Path]:
    """Run the fully offline, project-owned WeatherDC reference fixture."""

    output_dir = Path(output_dir).resolve()
    config, climate, telemetry = _materialize_small_study(output_dir)
    base = BenchmarkRunner().run(config)
    reference, reference_metrics = _reference_predictions(base, climate, telemetry)
    combined = PredictionFrame.from_pandas(
        pd.concat(
            [base.predictions.to_pandas(), reference.to_pandas()],
            ignore_index=True,
        )
    )
    metrics = dict(base.metrics)
    metrics["cooling_power"] = reference_metrics
    prediction_metrics = cast(dict[str, object], metrics["predictions"])
    split_metrics = cast(dict[str, object], prediction_metrics["split-000"])
    split_metrics["weatherdc--split-000"] = {"point": reference_metrics["weatherdc"]}
    split_metrics["weatherdc-persistence--split-000"] = {"point": reference_metrics["persistence"]}
    input_hashes = dict(base.input_hashes)
    input_hashes["weatherdc_reference_implementation"] = _sha256(Path(__file__))
    result = replace(
        base,
        predictions=combined,
        metrics=metrics,
        input_hashes=input_hashes,
    )
    run_dir = ArtifactWriter().write(result, output_dir)
    return result, run_dir


class _ConversionAdapter(Protocol):
    def download(self, cache_dir: Path, manifest: SourceManifest) -> DownloadManifest: ...

    def load(self, cache_dir: Path) -> tuple[ClimateForecastFrame, DCTelemetryFrame]: ...


def _observation_rows(climate: ClimateForecastFrame) -> pd.DataFrame:
    rows = climate.to_pandas()
    observation_times = (rows["issue_time"] == rows["available_at"]) & (
        rows["available_at"] == rows["valid_time"]
    )
    if not bool(observation_times.all()) or set(rows["source"]) != {"weatherdc:hii-observation"}:
        raise ConfigurationError(
            "Full conversion requires observation-only climate rows with "
            "issue_time = available_at = valid_time"
        )
    return rows


def prepare_conversion(
    cache_dir: Path,
    *,
    adapter: _ConversionAdapter | None = None,
    manifest: SourceManifest = WEATHERDC_SOURCE_MANIFEST,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[Path, float]:
    """Download and convert verified observations without creating a benchmark."""

    started = clock()
    cache_dir = Path(cache_dir).resolve()
    raw_dir = cache_dir / "raw"
    study_dir = cache_dir / "study"
    active_adapter = adapter if adapter is not None else WeatherDCAdapter()
    active_adapter.download(raw_dir, manifest)
    climate, telemetry = active_adapter.load(raw_dir)
    climate_rows = _observation_rows(climate)
    study_dir.mkdir(parents=True, exist_ok=True)
    climate_path = study_dir / "climate.csv"
    telemetry_path = study_dir / "telemetry.csv"
    climate_rows.to_csv(climate_path, index=False, lineterminator="\n")
    telemetry.to_pandas().to_csv(telemetry_path, index=False, lineterminator="\n")
    elapsed = clock() - started
    conversion_manifest: dict[str, object] = {
        "mode": "conversion_only",
        "elapsed_seconds": float(elapsed),
        "benchmark_produced": False,
        "workload_produced": False,
        "observation_time_semantics": "issue_time = available_at = valid_time",
        "sources": [
            {
                "name": item.name,
                "url": item.url,
                "bytes": item.bytes,
                "sha256": item.sha256,
            }
            for item in manifest.items
        ],
        "outputs": {
            climate_path.name: _sha256(climate_path),
            telemetry_path.name: _sha256(telemetry_path),
        },
    }
    _write_yaml(study_dir / "conversion-manifest.yaml", conversion_manifest)
    return study_dir, elapsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run small mode or convert WeatherDC observations")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--small", action="store_true", help="run the offline CC0 fixture")
    mode.add_argument(
        "--full",
        action="store_true",
        help="verified upstream conversion only; produces no benchmark or workload",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "weatherdc-small")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "climadc" / "weatherdc")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.small:
        _, run_dir = run_small(args.output_dir)
        print(run_dir)
        return
    study_dir, elapsed = prepare_conversion(args.cache_dir)
    print(f"Converted upstream observations to {study_dir}")
    print(f"Download and conversion runtime: {elapsed:.1f} seconds")
    print(
        "Upstream files are not relicensed by ClimaDC; review the energydata and HII terms "
        "before reuse. Full weather-model retraining is intentionally not performed."
    )


if __name__ == "__main__":
    main()
