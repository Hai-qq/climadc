from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from climadc.errors import ConfigurationError, ContractError
from climadc.replay import ReplayConfig, TemperatureSensitivePUEModel


def test_replay_config_normalizes_finite_engineering_parameters() -> None:
    config = ReplayConfig(
        site_id="dc-1",
        horizon=pd.Timedelta(hours=4),
        interval=pd.Timedelta(hours=1),
        it_capacity_kw=4,
        fixed_it_power_kw=1,
        cost_weight=2,
        carbon_weight=3,
        demand_charge_per_kw=4,
    )

    assert config.it_capacity_kw == 4.0
    assert config.fixed_it_power_kw == 1.0
    assert config.slot_count == 4
    with pytest.raises(FrozenInstanceError):
        config.site_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("site_id", "", "site_id"),
        ("horizon", pd.Timedelta(0), "horizon"),
        ("interval", pd.Timedelta(0), "interval"),
        ("it_capacity_kw", 0.0, "it_capacity_kw"),
        ("fixed_it_power_kw", -1.0, "fixed_it_power_kw"),
        ("fixed_it_power_kw", 5.0, "must not exceed"),
        ("cost_weight", -1.0, "cost_weight"),
        ("carbon_weight", float("inf"), "carbon_weight"),
        ("demand_charge_per_kw", -1.0, "demand_charge_per_kw"),
        ("risk_quantile", 0.5, "risk_quantile"),
        ("risk_quantile", 1.0, "risk_quantile"),
        ("tolerance_kwh", 0.0, "tolerance_kwh"),
    ],
)
def test_replay_config_rejects_invalid_parameters(field: str, value: object, message: str) -> None:
    values: dict[str, object] = {
        "site_id": "dc-1",
        "horizon": pd.Timedelta(hours=4),
        "interval": pd.Timedelta(hours=1),
        "it_capacity_kw": 4.0,
        "fixed_it_power_kw": 1.0,
        "cost_weight": 1.0,
        "carbon_weight": 1.0,
        "demand_charge_per_kw": 0.0,
        "tolerance_kwh": 1e-7,
    }
    values[field] = value

    with pytest.raises(ConfigurationError, match=message):
        ReplayConfig(**values)  # type: ignore[arg-type]


def test_replay_config_requires_integral_slot_count_and_nonzero_joint_objective() -> None:
    with pytest.raises(ConfigurationError, match="integer multiple"):
        ReplayConfig(
            site_id="dc-1",
            horizon=pd.Timedelta(minutes=90),
            interval=pd.Timedelta(hours=1),
            it_capacity_kw=4,
        )

    with pytest.raises(ConfigurationError, match="joint objective"):
        ReplayConfig(
            site_id="dc-1",
            horizon=pd.Timedelta(hours=4),
            interval=pd.Timedelta(hours=1),
            it_capacity_kw=4,
            cost_weight=0,
            carbon_weight=0,
        )


def test_temperature_sensitive_pue_model_is_monotonic_bounded_and_copy_safe() -> None:
    model = TemperatureSensitivePUEModel(
        reference_temperature_c=18,
        base_pue=1.2,
        slope_per_degree_c=0.02,
        min_pue=1.1,
        max_pue=1.5,
    )
    temperatures = np.array([0.0, 18.0, 25.0, 100.0])

    result = model.pue(temperatures)

    np.testing.assert_allclose(result, [1.1, 1.2, 1.34, 1.5])
    assert np.all(np.diff(result) >= 0.0)
    assert not np.shares_memory(result, temperatures)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reference_temperature_c", float("nan")),
        ("base_pue", 0.9),
        ("slope_per_degree_c", -0.01),
        ("min_pue", 0.9),
        ("max_pue", 0.9),
        ("max_pue", float("inf")),
    ],
)
def test_temperature_sensitive_pue_model_rejects_invalid_parameters(
    field: str, value: object
) -> None:
    values: dict[str, object] = {
        "reference_temperature_c": 18.0,
        "base_pue": 1.2,
        "slope_per_degree_c": 0.02,
        "min_pue": 1.1,
        "max_pue": 2.0,
    }
    values[field] = value

    with pytest.raises(ConfigurationError):
        TemperatureSensitivePUEModel(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "temperatures",
    [
        np.array([1.0, float("nan")]),
        np.array([[1.0, 2.0]]),
        np.array([], dtype=float),
    ],
)
def test_temperature_sensitive_pue_model_rejects_invalid_temperature_vectors(
    temperatures: np.ndarray,
) -> None:
    with pytest.raises(ContractError, match="temperature"):
        TemperatureSensitivePUEModel().pue(temperatures)
