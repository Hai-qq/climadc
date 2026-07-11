from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from climadc.contracts.frames import ClimateForecastFrame
from climadc.errors import ConfigurationError


@dataclass(frozen=True)
class SliceAudit:
    """Sample-count audit for a named evaluation slice."""

    name: str
    sample_count: int
    minimum: int
    usable: bool


class SliceSizeWarning(UserWarning):
    """Structured warning emitted for a slice below its declared minimum."""

    def __init__(self, name: str, sample_count: int, minimum: int) -> None:
        self.name = name
        self.sample_count = sample_count
        self.minimum = minimum
        super().__init__(f"Slice {name!r} has {sample_count} samples, below minimum {minimum}")


def extreme_weather_mask(
    climate: ClimateForecastFrame,
    variable: str,
    quantile: float,
) -> pd.Series:
    """Return an aligned upper-tail mask for one climate variable."""

    if not isinstance(variable, str) or not variable.strip():
        raise ConfigurationError("variable must be a non-empty string")
    if (
        not isinstance(quantile, Real)
        or isinstance(quantile, bool)
        or not math.isfinite(float(quantile))
        or not 0.0 < float(quantile) < 1.0
    ):
        raise ConfigurationError("quantile must be a finite number strictly inside (0, 1)")

    frame = climate.to_pandas()
    selected = frame["variable"] == variable
    values = frame.loc[selected, "value"].to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        raise ConfigurationError(f"No climate rows with finite values for variable {variable!r}")
    threshold = float(np.quantile(finite_values, float(quantile), method="higher"))
    mask = selected & frame["value"].ge(threshold) & frame["value"].map(np.isfinite)
    return pd.Series(mask.to_numpy(dtype=bool), index=frame.index, dtype=bool, name=variable)


def audit_slice(mask: pd.Series, name: str, minimum: int) -> SliceAudit:
    """Count a boolean slice and warn once when it is too small for evaluation."""

    if not isinstance(mask, pd.Series):
        raise ConfigurationError("mask must be a pandas Series")
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("name must be a non-empty string")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise ConfigurationError("minimum must be an integer greater than or equal to 1")
    if mask.isna().any() or not pd.api.types.is_bool_dtype(mask.dtype):
        raise ConfigurationError("mask must contain non-null boolean values")

    sample_count = int(mask.sum())
    usable = sample_count >= minimum
    audit = SliceAudit(name=name, sample_count=sample_count, minimum=minimum, usable=usable)
    if not usable:
        warnings.warn(SliceSizeWarning(name, sample_count, minimum), stacklevel=2)
    return audit
