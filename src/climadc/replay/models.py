from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from climadc.errors import ConfigurationError, ContractError


def _finite_float(value: object, *, field: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(float(value)):
        raise ConfigurationError(f"{field} must be a finite real number")
    return float(value)


def _positive_timedelta(value: object, *, field: str) -> pd.Timedelta:
    if not isinstance(value, pd.Timedelta) or pd.isna(value) or value <= pd.Timedelta(0):
        raise ConfigurationError(f"{field} must be a positive pandas Timedelta")
    return value


@dataclass(frozen=True)
class ReplayConfig:
    """Physical and economic settings for one replay decision window."""

    site_id: str
    horizon: pd.Timedelta
    interval: pd.Timedelta
    it_capacity_kw: float
    fixed_it_power_kw: float = 0.0
    cost_weight: float = 1.0
    carbon_weight: float = 1.0
    demand_charge_per_kw: float = 0.0
    risk_quantile: float | None = None
    tolerance_kwh: float = 1e-7
    temperature_variable: str = "air_temperature"
    weather_metric: str = "air_temperature"

    def __post_init__(self) -> None:
        if not isinstance(self.site_id, str) or not self.site_id.strip():
            raise ConfigurationError("site_id must be a non-empty string")
        horizon = _positive_timedelta(self.horizon, field="horizon")
        interval = _positive_timedelta(self.interval, field="interval")
        if horizon % interval != pd.Timedelta(0):
            raise ConfigurationError("horizon must be an integer multiple of interval")

        it_capacity_kw = _finite_float(self.it_capacity_kw, field="it_capacity_kw")
        fixed_it_power_kw = _finite_float(self.fixed_it_power_kw, field="fixed_it_power_kw")
        cost_weight = _finite_float(self.cost_weight, field="cost_weight")
        carbon_weight = _finite_float(self.carbon_weight, field="carbon_weight")
        demand_charge_per_kw = _finite_float(
            self.demand_charge_per_kw, field="demand_charge_per_kw"
        )
        risk_quantile = (
            None
            if self.risk_quantile is None
            else _finite_float(self.risk_quantile, field="risk_quantile")
        )
        tolerance_kwh = _finite_float(self.tolerance_kwh, field="tolerance_kwh")

        if it_capacity_kw <= 0.0:
            raise ConfigurationError("it_capacity_kw must be positive")
        if fixed_it_power_kw < 0.0:
            raise ConfigurationError("fixed_it_power_kw must be nonnegative")
        if fixed_it_power_kw > it_capacity_kw:
            raise ConfigurationError("fixed_it_power_kw must not exceed it_capacity_kw")
        if cost_weight < 0.0:
            raise ConfigurationError("cost_weight must be nonnegative")
        if carbon_weight < 0.0:
            raise ConfigurationError("carbon_weight must be nonnegative")
        if cost_weight == 0.0 and carbon_weight == 0.0:
            raise ConfigurationError("joint objective requires a nonzero cost or carbon weight")
        if demand_charge_per_kw < 0.0:
            raise ConfigurationError("demand_charge_per_kw must be nonnegative")
        if risk_quantile is not None and not 0.5 < risk_quantile < 1.0:
            raise ConfigurationError("risk_quantile must be strictly inside (0.5, 1)")
        if tolerance_kwh <= 0.0:
            raise ConfigurationError("tolerance_kwh must be positive")
        for field, value in (
            ("temperature_variable", self.temperature_variable),
            ("weather_metric", self.weather_metric),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{field} must be a non-empty string")

        object.__setattr__(self, "site_id", self.site_id.strip())
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "it_capacity_kw", it_capacity_kw)
        object.__setattr__(self, "fixed_it_power_kw", fixed_it_power_kw)
        object.__setattr__(self, "cost_weight", cost_weight)
        object.__setattr__(self, "carbon_weight", carbon_weight)
        object.__setattr__(self, "demand_charge_per_kw", demand_charge_per_kw)
        object.__setattr__(self, "risk_quantile", risk_quantile)
        object.__setattr__(self, "tolerance_kwh", tolerance_kwh)
        object.__setattr__(self, "temperature_variable", self.temperature_variable.strip())
        object.__setattr__(self, "weather_metric", self.weather_metric.strip())

    @property
    def slot_count(self) -> int:
        return int(self.horizon // self.interval)


@runtime_checkable
class FacilityEnergyModel(Protocol):
    """Convert an ambient-temperature profile into a facility PUE profile."""

    def pue(self, ambient_temperature_c: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class TemperatureSensitivePUEModel:
    """Bounded linear PUE model suitable for deterministic replay examples."""

    reference_temperature_c: float = 18.0
    base_pue: float = 1.2
    slope_per_degree_c: float = 0.02
    min_pue: float = 1.1
    max_pue: float = 2.0

    def __post_init__(self) -> None:
        reference_temperature_c = _finite_float(
            self.reference_temperature_c, field="reference_temperature_c"
        )
        base_pue = _finite_float(self.base_pue, field="base_pue")
        slope_per_degree_c = _finite_float(self.slope_per_degree_c, field="slope_per_degree_c")
        min_pue = _finite_float(self.min_pue, field="min_pue")
        max_pue = _finite_float(self.max_pue, field="max_pue")
        if min_pue < 1.0:
            raise ConfigurationError("min_pue must be at least 1")
        if not min_pue <= base_pue <= max_pue:
            raise ConfigurationError("PUE bounds require min_pue <= base_pue <= max_pue")
        if min_pue >= max_pue:
            raise ConfigurationError("max_pue must be greater than min_pue")
        if slope_per_degree_c < 0.0:
            raise ConfigurationError("slope_per_degree_c must be nonnegative")
        object.__setattr__(self, "reference_temperature_c", reference_temperature_c)
        object.__setattr__(self, "base_pue", base_pue)
        object.__setattr__(self, "slope_per_degree_c", slope_per_degree_c)
        object.__setattr__(self, "min_pue", min_pue)
        object.__setattr__(self, "max_pue", max_pue)

    def pue(self, ambient_temperature_c: np.ndarray) -> np.ndarray:
        try:
            temperatures = np.asarray(ambient_temperature_c, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ContractError("temperature input must be a numeric vector") from exc
        if temperatures.ndim != 1 or temperatures.size == 0:
            raise ContractError("temperature input must be a non-empty one-dimensional vector")
        if not np.isfinite(temperatures).all():
            raise ContractError("temperature input must contain only finite values")
        values = self.base_pue + self.slope_per_degree_c * (
            temperatures - self.reference_temperature_c
        )
        result: np.ndarray = np.asarray(
            np.clip(values, self.min_pue, self.max_pue), dtype=float
        ).copy()
        return result
