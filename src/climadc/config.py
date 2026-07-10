from __future__ import annotations

from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from climadc.errors import ConfigurationError


def _validate_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError(f"Unknown IANA timezone: {value}") from exc
    return value


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


class InputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    format: Literal["csv", "parquet", "xarray"]
    timezone: str
    card: Path
    column_map: dict[str, str] = Field(default_factory=dict)

    _timezone_is_iana = field_validator("timezone")(_validate_timezone)

    def resolve_paths(self, base_dir: Path) -> InputConfig:
        return self.model_copy(
            update={
                "path": _resolve_path(self.path, base_dir),
                "card": _resolve_path(self.card, base_dir),
            }
        )


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["blocked", "rolling"]
    min_train: int = Field(ge=2)
    calibration_size: int = Field(ge=1)
    test_size: int = Field(ge=1)
    step: int = Field(ge=1)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["persistence", "seasonal", "climatology", "linear", "lightgbm"]
    model_id: str = Field(min_length=1)
    params: dict[str, object] = Field(default_factory=dict)


class DecisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    flexible_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    max_shift_multiplier: float = Field(default=2.0, ge=1.0)
    peak_penalty: float = Field(default=4.0, ge=0.0)
    risk_penalty: float = Field(default=0.25, ge=0.0)


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_id: str = Field(min_length=1)
    horizon: str
    climate: InputConfig
    telemetry: InputConfig
    workload: InputConfig | None = None
    backtest: BacktestConfig
    models: list[ModelConfig] = Field(min_length=1)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    output_dir: Path = Path("runs")

    @field_validator("horizon")
    @classmethod
    def horizon_is_positive_duration(cls, value: str) -> str:
        try:
            duration = pd.Timedelta(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("horizon must be a pandas Timedelta string") from exc
        if pd.isna(duration) or duration <= pd.Timedelta(0):
            raise ValueError("horizon must be positive")
        return value

    @classmethod
    def from_yaml(cls, path: Path) -> StudyConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            cfg = cls.model_validate(raw)
        except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
            raise ConfigurationError(f"Invalid study config {path}: {exc}") from exc
        return cfg.resolve_paths(path.parent)

    def resolve_paths(self, base_dir: Path) -> StudyConfig:
        updates: dict[str, object] = {
            "climate": self.climate.resolve_paths(base_dir),
            "telemetry": self.telemetry.resolve_paths(base_dir),
            "output_dir": _resolve_path(self.output_dir, base_dir),
        }
        if self.workload is not None:
            updates["workload"] = self.workload.resolve_paths(base_dir)
        return self.model_copy(update=updates)
