from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from climadc.errors import ConfigurationError
from climadc.replay.models import ReplayConfig, TemperatureSensitivePUEModel

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _resolve(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _utc_datetime(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None or value.utcoffset() != pd.Timedelta(0):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value


class ReplayInputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    format: Literal["csv", "parquet"] = "csv"
    timezone: str = "UTC"

    @field_validator("timezone")
    @classmethod
    def timezone_is_iana(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError(f"Unknown IANA timezone: {value}") from exc
        return value

    def resolve_path(self, base_dir: Path) -> ReplayInputConfig:
        return self.model_copy(update={"path": _resolve(self.path, base_dir)})


class ReplayInputsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    climate_forecast: ReplayInputConfig
    actual_weather: ReplayInputConfig
    grid_signals: ReplayInputConfig
    workload: ReplayInputConfig

    def resolve_paths(self, base_dir: Path) -> ReplayInputsConfig:
        return self.model_copy(
            update={
                name: getattr(self, name).resolve_path(base_dir)
                for name in (
                    "climate_forecast",
                    "actual_weather",
                    "grid_signals",
                    "workload",
                )
            }
        )


class ReplaySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    site_id: NonEmptyText
    horizon: NonEmptyText
    interval: NonEmptyText
    it_capacity_kw: float = Field(gt=0.0)
    fixed_it_power_kw: float = Field(default=0.0, ge=0.0)
    cost_weight: float = Field(default=1.0, ge=0.0)
    carbon_weight: float = Field(default=1.0, ge=0.0)
    demand_charge_per_kw: float = Field(default=0.0, ge=0.0)
    risk_quantile: float | None = Field(default=None, gt=0.5, lt=1.0)
    tolerance_kwh: float = Field(default=1e-7, gt=0.0)
    temperature_variable: NonEmptyText = "air_temperature"
    weather_metric: NonEmptyText = "air_temperature"

    @model_validator(mode="after")
    def settings_are_consistent(self) -> ReplaySettings:
        try:
            horizon = pd.Timedelta(self.horizon)
            interval = pd.Timedelta(self.interval)
        except (TypeError, ValueError) as exc:
            raise ValueError("horizon and interval must be pandas Timedelta strings") from exc
        if pd.isna(horizon) or pd.isna(interval) or horizon <= pd.Timedelta(0):
            raise ValueError("horizon and interval must be positive")
        if interval <= pd.Timedelta(0) or horizon % interval != pd.Timedelta(0):
            raise ValueError("horizon must be an integer multiple of interval")
        if self.fixed_it_power_kw > self.it_capacity_kw:
            raise ValueError("fixed_it_power_kw must not exceed it_capacity_kw")
        if self.cost_weight == 0.0 and self.carbon_weight == 0.0:
            raise ValueError("joint objective requires a nonzero cost or carbon weight")
        return self

    def to_replay_config(self) -> ReplayConfig:
        return ReplayConfig(
            site_id=self.site_id,
            horizon=pd.Timedelta(self.horizon),
            interval=pd.Timedelta(self.interval),
            it_capacity_kw=self.it_capacity_kw,
            fixed_it_power_kw=self.fixed_it_power_kw,
            cost_weight=self.cost_weight,
            carbon_weight=self.carbon_weight,
            demand_charge_per_kw=self.demand_charge_per_kw,
            risk_quantile=self.risk_quantile,
            tolerance_kwh=self.tolerance_kwh,
            temperature_variable=self.temperature_variable,
            weather_metric=self.weather_metric,
        )


class TemperatureSensitivePUEConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["temperature_sensitive_pue"] = "temperature_sensitive_pue"
    reference_temperature_c: float = 18.0
    base_pue: float = 1.2
    slope_per_degree_c: float = Field(default=0.02, ge=0.0)
    min_pue: float = Field(default=1.1, ge=1.0)
    max_pue: float = Field(default=2.0, gt=1.0)

    def build(self) -> TemperatureSensitivePUEModel:
        return TemperatureSensitivePUEModel(
            reference_temperature_c=self.reference_temperature_c,
            base_pue=self.base_pue,
            slope_per_degree_c=self.slope_per_degree_c,
            min_pue=self.min_pue,
            max_pue=self.max_pue,
        )


class RollingReplaySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    periods: int = Field(ge=1)
    step: NonEmptyText

    @field_validator("step")
    @classmethod
    def step_is_positive(cls, value: str) -> str:
        try:
            step = pd.Timedelta(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("rolling step must be a pandas Timedelta string") from exc
        if pd.isna(step) or step <= pd.Timedelta(0):
            raise ValueError("rolling step must be positive")
        return value

    @property
    def step_timedelta(self) -> pd.Timedelta:
        return cast(pd.Timedelta, pd.Timedelta(self.step))


class ReplayStudyConfig(BaseModel):
    """Strict, path-resolved configuration for a local engineering replay."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    study_id: NonEmptyText
    decision_time: datetime
    inputs: ReplayInputsConfig
    source_manifest: Path
    replay: ReplaySettings
    rolling: RollingReplaySettings | None = None
    facility_model: TemperatureSensitivePUEConfig = Field(
        default_factory=TemperatureSensitivePUEConfig
    )
    assumptions: dict[str, object] = Field(default_factory=dict)
    limitations: list[NonEmptyText] = Field(default_factory=list)
    output_dir: Path = Path("replay-runs")

    @field_validator("decision_time")
    @classmethod
    def decision_time_is_utc(cls, value: datetime) -> datetime:
        return _utc_datetime(value, "decision_time")

    @model_validator(mode="after")
    def rolling_settings_match_replay_grid(self) -> ReplayStudyConfig:
        if self.rolling is None:
            return self
        step = self.rolling.step_timedelta
        interval = pd.Timedelta(self.replay.interval)
        horizon = pd.Timedelta(self.replay.horizon)
        if step % interval != pd.Timedelta(0):
            raise ValueError("rolling step must be an integer multiple of replay interval")
        if step > horizon:
            raise ValueError("rolling step must not exceed replay horizon")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> ReplayStudyConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            config = cls.model_validate(raw)
        except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
            raise ConfigurationError(f"Invalid replay config {path}: {exc}") from exc
        return config.resolve_paths(path.parent)

    def resolve_paths(self, base_dir: Path) -> ReplayStudyConfig:
        return self.model_copy(
            update={
                "inputs": self.inputs.resolve_paths(base_dir),
                "source_manifest": _resolve(self.source_manifest, base_dir),
                "output_dir": _resolve(self.output_dir, base_dir),
            }
        )

    @property
    def decision_timestamp(self) -> pd.Timestamp:
        return cast(pd.Timestamp, pd.Timestamp(self.decision_time).tz_convert("UTC"))

    def with_output_dir(self, output_dir: Path) -> ReplayStudyConfig:
        return self.model_copy(update={"output_dir": Path(output_dir).resolve()})
