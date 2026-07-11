from dataclasses import FrozenInstanceError
from typing import Any

import pandas as pd
import pytest

from climadc.contracts.frames import PredictionFrame, WorkloadFrame
from climadc.decision import DecisionConstraints, DecisionPolicy, DecisionResult, ShadowScheduler
from climadc.errors import ConfigurationError, ContractError


class UserDecisionPolicy:
    def solve(
        self,
        forecast: PredictionFrame,
        workload: WorkloadFrame,
        constraints: DecisionConstraints,
    ) -> DecisionResult:
        raise NotImplementedError


class MissingSolve:
    pass


def test_decision_policy_is_runtime_checkable_and_structural() -> None:
    assert isinstance(UserDecisionPolicy(), DecisionPolicy)
    assert isinstance(ShadowScheduler(), DecisionPolicy)
    assert not isinstance(MissingSolve(), DecisionPolicy)


def test_decision_constraints_defaults_are_frozen() -> None:
    constraints = DecisionConstraints()

    assert constraints == DecisionConstraints(0.1, 2.0, 4.0, 0.25)
    with pytest.raises(FrozenInstanceError):
        constraints.flexible_fraction = 0.2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("flexible_fraction", -0.01),
        ("flexible_fraction", 1.01),
        ("max_shift_multiplier", 0.99),
        ("peak_penalty", -0.01),
        ("risk_penalty", -0.01),
        ("flexible_fraction", True),
        ("max_shift_multiplier", False),
        ("peak_penalty", float("nan")),
        ("risk_penalty", float("inf")),
        ("risk_penalty", "0.25"),
    ],
)
def test_decision_constraints_reject_invalid_values(field: str, value: object) -> None:
    values: dict[str, Any] = {
        "flexible_fraction": 0.1,
        "max_shift_multiplier": 2.0,
        "peak_penalty": 4.0,
        "risk_penalty": 0.25,
    }
    values[field] = value

    with pytest.raises(ConfigurationError, match=field):
        DecisionConstraints(**values)


def test_decision_result_is_frozen_and_isolates_constructor_inputs() -> None:
    schedule = pd.DataFrame({"total_after": [1.0]})
    metrics = {"solver_status": 0.0}

    result = DecisionResult(schedule, True, ["note"], metrics)  # type: ignore[arg-type]
    schedule.loc[0, "total_after"] = 99.0
    metrics["solver_status"] = 99.0

    assert result.schedule.loc[0, "total_after"] == 1.0
    assert result.violations == ("note",)
    assert result.metrics == {"solver_status": 0.0}
    with pytest.raises(FrozenInstanceError):
        result.feasible = False  # type: ignore[misc]


@pytest.mark.parametrize("metric", [True, float("nan"), float("inf"), "zero"])
def test_decision_result_requires_finite_float_metrics(metric: object) -> None:
    with pytest.raises(ContractError, match="metrics"):
        DecisionResult(pd.DataFrame(), False, (), {"solver_status": metric})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"schedule": "not-a-frame"},
        {"feasible": 1},
        {"violations": (1,)},
        {"metrics": {1: 0.0}},
    ],
)
def test_decision_result_rejects_malformed_fields(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "schedule": pd.DataFrame(),
        "feasible": False,
        "violations": (),
        "metrics": {},
    }
    values.update(kwargs)

    with pytest.raises(ContractError, match="DecisionResult"):
        DecisionResult(**values)  # type: ignore[arg-type]
