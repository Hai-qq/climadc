from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, cast
import warnings
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


class MonetizedObjectiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["1"] = "1"
    mode: Literal["monetized"] = "monetized"
    carbon_price_currency_per_tco2e: float = Field(ge=0.0)
    demand_charge_per_kw: float = Field(default=0.0, ge=0.0)


class EpsilonConstraintObjectiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["1"] = "1"
    mode: Literal["epsilon_constraint"] = "epsilon_constraint"
    emissions_upper_bound_kgco2e: float | None = Field(default=None, gt=0.0)
    peak_upper_bound_kw: float | None = Field(default=None, gt=0.0)
    demand_charge_per_kw: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def at_least_one_constraint(self) -> EpsilonConstraintObjectiveConfig:
        if self.emissions_upper_bound_kgco2e is None and self.peak_upper_bound_kw is None:
            raise ValueError("epsilon_constraint requires an emissions and/or peak upper bound")
        return self


class ParetoAnalysisObjectiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["1"] = "1"
    mode: Literal["pareto_analysis"] = "pareto_analysis"
    carbon_prices_currency_per_tco2e: list[float] = Field(min_length=2)
    demand_charge_per_kw: float = Field(default=0.0, ge=0.0)

    @field_validator("carbon_prices_currency_per_tco2e")
    @classmethod
    def prices_are_public_fixed_points(cls, values: list[float]) -> list[float]:
        if any(not 0.0 <= value < float("inf") for value in values):
            raise ValueError("pareto carbon prices must be finite and nonnegative")
        if len(values) != len(set(values)):
            raise ValueError("pareto carbon prices must be unique")
        if values != sorted(values):
            raise ValueError("pareto carbon prices must use stable ascending order")
        return values


ObjectiveConfig = Annotated[
    MonetizedObjectiveConfig | EpsilonConstraintObjectiveConfig | ParetoAnalysisObjectiveConfig,
    Field(discriminator="mode"),
]


class ReplaySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    site_id: NonEmptyText
    horizon: NonEmptyText
    interval: NonEmptyText
    it_capacity_kw: float = Field(gt=0.0)
    fixed_it_power_kw: float = Field(default=0.0, ge=0.0)
    objective: ObjectiveConfig | None = None
    cost_weight: float | None = Field(default=None, ge=0.0)
    carbon_weight: float | None = Field(default=None, ge=0.0)
    demand_charge_per_kw: float | None = Field(default=None, ge=0.0)
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
        legacy_fields = (self.cost_weight, self.carbon_weight, self.demand_charge_per_kw)
        if self.objective is not None and any(value is not None for value in legacy_fields):
            raise ValueError(
                "objective must not be combined with legacy cost_weight, carbon_weight, or "
                "demand_charge_per_kw"
            )
        if (
            isinstance(self.objective, ParetoAnalysisObjectiveConfig)
            and self.risk_quantile is not None
        ):
            raise ValueError("pareto_analysis cannot be combined with risk_quantile")
        cost_weight = 1.0 if self.cost_weight is None else self.cost_weight
        carbon_weight = 1.0 if self.carbon_weight is None else self.carbon_weight
        if self.objective is None and cost_weight == 0.0 and carbon_weight == 0.0:
            raise ValueError("joint objective requires a nonzero cost or carbon weight")
        return self

    def to_replay_config(self) -> ReplayConfig:
        objective_mode: Literal[
            "legacy_unscaled", "monetized", "epsilon_constraint", "pareto_analysis"
        ]
        if self.objective is None:
            warnings.warn(
                (
                    "cost_weight/carbon_weight is deprecated and dimensionally unscaled; "
                    "migrate to replay.objective"
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            objective_mode = "legacy_unscaled"
            carbon_price = 0.0
            emissions_bound = None
            peak_bound = None
            pareto_prices: tuple[float, ...] = ()
            demand_charge = 0.0 if self.demand_charge_per_kw is None else self.demand_charge_per_kw
        else:
            objective_mode = self.objective.mode
            demand_charge = self.objective.demand_charge_per_kw
            carbon_price = (
                self.objective.carbon_price_currency_per_tco2e
                if isinstance(self.objective, MonetizedObjectiveConfig)
                else 0.0
            )
            emissions_bound = (
                self.objective.emissions_upper_bound_kgco2e
                if isinstance(self.objective, EpsilonConstraintObjectiveConfig)
                else None
            )
            peak_bound = (
                self.objective.peak_upper_bound_kw
                if isinstance(self.objective, EpsilonConstraintObjectiveConfig)
                else None
            )
            pareto_prices = (
                tuple(self.objective.carbon_prices_currency_per_tco2e)
                if isinstance(self.objective, ParetoAnalysisObjectiveConfig)
                else ()
            )
        return ReplayConfig(
            site_id=self.site_id,
            horizon=pd.Timedelta(self.horizon),
            interval=pd.Timedelta(self.interval),
            it_capacity_kw=self.it_capacity_kw,
            fixed_it_power_kw=self.fixed_it_power_kw,
            cost_weight=1.0 if self.cost_weight is None else self.cost_weight,
            carbon_weight=1.0 if self.carbon_weight is None else self.carbon_weight,
            demand_charge_per_kw=demand_charge,
            objective_mode=objective_mode,
            carbon_price_currency_per_tco2e=carbon_price,
            emissions_upper_bound_kgco2e=emissions_bound,
            peak_upper_bound_kw=peak_bound,
            pareto_carbon_prices_currency_per_tco2e=pareto_prices,
            risk_quantile=self.risk_quantile,
            tolerance_kwh=self.tolerance_kwh,
            temperature_variable=self.temperature_variable,
            weather_metric=self.weather_metric,
        )

    def objective_payload(self) -> dict[str, object]:
        """Return the versioned objective contract written into evidence artifacts."""
        if self.objective is not None:
            return cast(dict[str, object], self.objective.model_dump(mode="json"))
        return {
            "version": "legacy-1",
            "mode": "legacy_unscaled",
            "cost_weight": 1.0 if self.cost_weight is None else self.cost_weight,
            "carbon_weight": 1.0 if self.carbon_weight is None else self.carbon_weight,
            "demand_charge_per_kw": (
                0.0 if self.demand_charge_per_kw is None else self.demand_charge_per_kw
            ),
            "dimensionally_unscaled": True,
            "deprecated": True,
        }


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
        if isinstance(self.replay.objective, ParetoAnalysisObjectiveConfig):
            raise ValueError(
                "pareto_analysis is currently limited to single-window replay; "
                "use one explicit monetized objective for rolling replay"
            )
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
