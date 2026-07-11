from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Protocol, runtime_checkable

import pandas as pd

from climadc.contracts.frames import PredictionFrame, WorkloadFrame
from climadc.errors import ConfigurationError, ContractError


def _finite_float(value: object, *, field: str, error_type: type[Exception]) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(float(value)):
        raise error_type(f"{field} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class DecisionConstraints:
    flexible_fraction: float = 0.1
    max_shift_multiplier: float = 2.0
    peak_penalty: float = 4.0
    risk_penalty: float = 0.25

    def __post_init__(self) -> None:
        flexible_fraction = _finite_float(
            self.flexible_fraction,
            field="flexible_fraction",
            error_type=ConfigurationError,
        )
        max_shift_multiplier = _finite_float(
            self.max_shift_multiplier,
            field="max_shift_multiplier",
            error_type=ConfigurationError,
        )
        peak_penalty = _finite_float(
            self.peak_penalty,
            field="peak_penalty",
            error_type=ConfigurationError,
        )
        risk_penalty = _finite_float(
            self.risk_penalty,
            field="risk_penalty",
            error_type=ConfigurationError,
        )
        if not 0.0 <= flexible_fraction <= 1.0:
            raise ConfigurationError("flexible_fraction must be inside [0, 1]")
        if max_shift_multiplier < 1.0:
            raise ConfigurationError("max_shift_multiplier must be at least 1")
        if peak_penalty < 0.0:
            raise ConfigurationError("peak_penalty must be nonnegative")
        if risk_penalty < 0.0:
            raise ConfigurationError("risk_penalty must be nonnegative")
        object.__setattr__(self, "flexible_fraction", flexible_fraction)
        object.__setattr__(self, "max_shift_multiplier", max_shift_multiplier)
        object.__setattr__(self, "peak_penalty", peak_penalty)
        object.__setattr__(self, "risk_penalty", risk_penalty)


@dataclass(frozen=True)
class DecisionResult:
    schedule: pd.DataFrame
    feasible: bool
    violations: tuple[str, ...]
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, pd.DataFrame):
            raise ContractError("DecisionResult schedule must be a pandas DataFrame")
        if not isinstance(self.feasible, bool):
            raise ContractError("DecisionResult feasible must be bool")
        violations = tuple(self.violations)
        if any(not isinstance(violation, str) for violation in violations):
            raise ContractError("DecisionResult violations must contain only strings")
        metrics: dict[str, float] = {}
        for key, value in self.metrics.items():
            if not isinstance(key, str):
                raise ContractError("DecisionResult metrics keys must be strings")
            metrics[key] = _finite_float(
                value,
                field=f"DecisionResult metrics[{key!r}]",
                error_type=ContractError,
            )
        object.__setattr__(self, "schedule", self.schedule.copy(deep=True))
        object.__setattr__(self, "violations", violations)
        object.__setattr__(self, "metrics", metrics)


@runtime_checkable
class DecisionPolicy(Protocol):
    def solve(
        self,
        forecast: PredictionFrame,
        workload: WorkloadFrame,
        constraints: DecisionConstraints,
    ) -> DecisionResult: ...
