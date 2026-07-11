from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from climadc.errors import ConfigurationError

_START = pd.Timestamp("2026-01-01 00:00:00+00:00")
_PERIODS = 96


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csvs(directory: Path) -> tuple[Path, Path, Path]:
    times = pd.date_range(_START, periods=_PERIODS, freq="h")
    hours = np.arange(_PERIODS, dtype=float)
    rng = np.random.default_rng(20260710)
    temperature = 24.0 + 5.0 * np.sin(2.0 * np.pi * (hours % 24.0) / 24.0)
    total_power = (
        420.0
        + 3.0 * temperature
        + 12.0 * np.cos(2.0 * np.pi * (hours % 24.0) / 24.0)
        + rng.normal(0.0, 0.25, _PERIODS)
    )

    climate = pd.DataFrame(
        {
            "site_id": "dc-demo",
            "issue_time": _START.isoformat(),
            "available_at": _START.isoformat(),
            "valid_time": [(timestamp + pd.Timedelta("1h")).isoformat() for timestamp in times],
            "variable": "air_temperature",
            "value": np.round(temperature, 6),
            "unit": "degC",
            "source": "climadc-synthetic",
            "quantile": pd.NA,
            "member": pd.NA,
        }
    )
    telemetry = pd.DataFrame(
        {
            "site_id": "dc-demo",
            "device_id": "main-meter",
            "event_time": [timestamp.isoformat() for timestamp in times],
            "available_at": [timestamp.isoformat() for timestamp in times],
            "metric": "total_power",
            "value": np.round(total_power, 6),
            "unit": "kW",
            "quality": "observed",
        }
    )
    workload = pd.DataFrame(
        {
            "job_id": [f"batch-{position:03d}" for position in range(_PERIODS)],
            "site_id": "dc-demo",
            "event_time": [timestamp.isoformat() for timestamp in times],
            "available_at": [timestamp.isoformat() for timestamp in times],
            "deadline": [(timestamp + pd.Timedelta("4h")).isoformat() for timestamp in times],
            "resource_type": "compute_energy",
            "demand": np.round(8.0 + 2.0 * ((hours % 6.0) / 5.0), 6),
            "unit": "kWh",
            "flexible_fraction": 0.5,
        }
    )

    paths = (
        directory / "climate.csv",
        directory / "telemetry.csv",
        directory / "workload.csv",
    )
    for frame, path in zip((climate, telemetry, workload), paths, strict=True):
        frame.to_csv(path, index=False, lineterminator="\n")
    return paths


def _card_payload(
    name: str,
    digest: str,
    time_start: pd.Timestamp,
    time_end: pd.Timestamp,
) -> dict[str, object]:
    return {
        "name": name,
        "site": {
            "site_id": "dc-demo",
            "latitude": 0.0,
            "longitude": 0.0,
            "timezone": "UTC",
        },
        "source": {
            "provider": "ClimaDC project",
            "url": "https://github.com/Hai-qq/climadc",
            "license": "Apache-2.0",
            "redistribution_constraints": None,
        },
        "sha256": digest,
        "schema_version": "1.0",
        "time_start": time_start.isoformat(),
        "time_end": time_end.isoformat(),
        "sampling_frequency": "1h",
        "known_missing": [],
        "spatial_mismatch": [],
        "quality_limitations": [
            "Deterministic project-owned synthetic formula; not representative of operations."
        ],
    }


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
        newline="\n",
    )


def scaffold_study(directory: Path) -> Path:
    """Create one deterministic, fully local benchmark study."""

    directory = Path(directory)
    try:
        if directory.exists() and any(directory.iterdir()):
            raise ConfigurationError(f"Study directory is not empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        climate_path, telemetry_path, workload_path = _write_csvs(directory)
        observed_end = _START + pd.Timedelta(hours=_PERIODS - 1)
        inputs = (
            (
                "Synthetic climate",
                climate_path,
                directory / "climate-card.yaml",
                _START + pd.Timedelta("1h"),
                observed_end + pd.Timedelta("1h"),
            ),
            (
                "Synthetic telemetry",
                telemetry_path,
                directory / "telemetry-card.yaml",
                _START,
                observed_end,
            ),
            (
                "Synthetic workload",
                workload_path,
                directory / "workload-card.yaml",
                _START,
                observed_end,
            ),
        )
        for name, data_path, card_path, time_start, time_end in inputs:
            _write_yaml(
                card_path,
                _card_payload(name, _sha256(data_path), time_start, time_end),
            )

        config: dict[str, object] = {
            "study_id": "climadc-synthetic-demo",
            "horizon": "4h",
            "climate": {
                "path": "climate.csv",
                "format": "csv",
                "timezone": "UTC",
                "card": "climate-card.yaml",
                "column_map": {},
            },
            "telemetry": {
                "path": "telemetry.csv",
                "format": "csv",
                "timezone": "UTC",
                "card": "telemetry-card.yaml",
                "column_map": {},
            },
            "workload": {
                "path": "workload.csv",
                "format": "csv",
                "timezone": "UTC",
                "card": "workload-card.yaml",
                "column_map": {},
            },
            "backtest": {
                "strategy": "blocked",
                "min_train": 72,
                "calibration_size": 12,
                "test_size": 8,
                "step": 8,
            },
            "models": [
                {
                    "kind": "persistence",
                    "model_id": "persistence",
                    "params": {"target": "total_power"},
                },
                {
                    "kind": "seasonal",
                    "model_id": "seasonal",
                    "params": {"target": "total_power", "period": "24h"},
                },
                {
                    "kind": "climatology",
                    "model_id": "climatology",
                    "params": {"target": "total_power", "group_by": ["site_id", "hour"]},
                },
                {
                    "kind": "linear",
                    "model_id": "linear",
                    "params": {
                        "target": "total_power",
                        "features": ["hour", "dayofweek", "elapsed_hours"],
                    },
                },
            ],
            "decision": {
                "enabled": True,
                "flexible_fraction": 0.5,
                "max_shift_multiplier": 2.0,
                "peak_penalty": 4.0,
                "risk_penalty": 0.25,
            },
            "output_dir": "runs",
        }
        config_path = directory / "study.yaml"
        _write_yaml(config_path, config)
        return config_path
    except ConfigurationError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigurationError(f"Unable to initialize study {directory}: {exc}") from exc
