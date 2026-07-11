from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import yaml

from climadc.adapters.weatherdc import WEATHERDC_SOURCE_MANIFEST, WeatherDCAdapter
from climadc.benchmark import BenchmarkRunner, RunResult
from climadc.config import InputConfig, StudyConfig
from climadc.contracts.frames import PREDICTION_COLUMNS, PredictionFrame
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
    targets = cast(pd.Series, target_rows.set_index("event_time")["value"])
    train_index = train_times.intersection(features.index).intersection(targets.index)
    test_index = test_times.intersection(features.index).intersection(targets.index)
    train_x = np.column_stack(
        [np.ones(len(train_index)), features.loc[train_index, ["temp", "humid", "solar"]]]
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        train_x, targets.loc[train_index].to_numpy(dtype=float), rcond=None
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
    actual = targets.loc[test_index].to_numpy(dtype=float)
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
    result = replace(base, predictions=combined, metrics=metrics)
    run_dir = ArtifactWriter().write(result, output_dir)
    return result, run_dir


def prepare_full(cache_dir: Path) -> tuple[Path, float]:
    """Download verified upstream files and convert them; no model training is implied."""

    started = time.monotonic()
    cache_dir = Path(cache_dir).resolve()
    raw_dir = cache_dir / "raw"
    study_dir = cache_dir / "study"
    WeatherDCAdapter().download(raw_dir, WEATHERDC_SOURCE_MANIFEST)
    climate, telemetry = WeatherDCAdapter().load(raw_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    climate.to_pandas().to_csv(study_dir / "climate.csv", index=False, lineterminator="\n")
    telemetry.to_pandas().to_csv(study_dir / "telemetry.csv", index=False, lineterminator="\n")
    return study_dir, time.monotonic() - started


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or prepare the WeatherDC reference study")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--small", action="store_true", help="run the offline CC0 fixture")
    mode.add_argument("--full", action="store_true", help="download and convert upstream data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "weatherdc")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "climadc" / "weatherdc")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.small:
        _, run_dir = run_small(args.output_dir)
        print(run_dir)
        return
    study_dir, elapsed = prepare_full(args.cache_dir)
    print(f"Converted upstream observations to {study_dir}")
    print(f"Download and conversion runtime: {elapsed:.1f} seconds")
    print(
        "Upstream files are not relicensed by ClimaDC; review the energydata and HII terms "
        "before reuse. Full weather-model retraining is intentionally not performed."
    )


if __name__ == "__main__":
    main()
