from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from climadc.evidence.verify import verify_run
from climadc.reference import packaged_study_path
from climadc.replay import ReplayArtifactWriter, ReplayStudyConfig, ReplayStudyRunner
from climadc.replay.config import (
    EpsilonConstraintObjectiveConfig,
    MonetizedObjectiveConfig,
    ParetoAnalysisObjectiveConfig,
    ReplaySettings,
)
from climadc.replay.models import pareto_policy_name


def _settings(**updates: object) -> ReplaySettings:
    payload: dict[str, object] = {
        "site_id": "dc-1",
        "horizon": "4h",
        "interval": "1h",
        "it_capacity_kw": 100.0,
    }
    payload.update(updates)
    return ReplaySettings.model_validate(payload)


def test_monetized_objective_converts_kgco2e_to_tco2e_before_pricing() -> None:
    settings = _settings(
        objective={
            "version": "1",
            "mode": "monetized",
            "carbon_price_currency_per_tco2e": 1000.0,
        }
    )
    config = settings.to_replay_config()

    assert isinstance(settings.objective, MonetizedObjectiveConfig)
    assert config.realized_objective(100.0, 50.0) == pytest.approx(150.0)
    assert settings.objective_payload() == {
        "version": "1",
        "mode": "monetized",
        "carbon_price_currency_per_tco2e": 1000.0,
        "demand_charge_per_kw": 0.0,
    }


def test_epsilon_constraint_requires_a_declared_physical_bound() -> None:
    with pytest.raises(ValueError, match="requires an emissions and/or peak upper bound"):
        _settings(objective={"version": "1", "mode": "epsilon_constraint"})

    settings = _settings(
        objective={
            "version": "1",
            "mode": "epsilon_constraint",
            "emissions_upper_bound_kgco2e": 250.0,
            "peak_upper_bound_kw": 90.0,
        }
    )
    config = settings.to_replay_config()

    assert isinstance(settings.objective, EpsilonConstraintObjectiveConfig)
    assert config.objective_mode == "epsilon_constraint"
    assert config.emissions_upper_bound_kgco2e == 250.0
    assert config.peak_upper_bound_kw == 90.0


def test_legacy_unscaled_objective_warns_and_cannot_mix_with_v1_objective() -> None:
    legacy = _settings(cost_weight=1.0, carbon_weight=2.0)
    with pytest.warns(DeprecationWarning, match="dimensionally unscaled"):
        config = legacy.to_replay_config()
    assert config.objective_mode == "legacy_unscaled"
    assert legacy.objective_payload()["deprecated"] is True

    with pytest.raises(ValueError, match="must not be combined"):
        _settings(
            objective={
                "version": "1",
                "mode": "monetized",
                "carbon_price_currency_per_tco2e": 1000.0,
            },
            cost_weight=1.0,
        )


def test_pareto_analysis_emits_each_declared_point_and_verified_artifacts(
    tmp_path: Path,
) -> None:
    base = ReplayStudyConfig.from_yaml(packaged_study_path())
    replay_payload = base.replay.model_dump(mode="python")
    replay_payload.update(
        {
            "objective": {
                "version": "1",
                "mode": "pareto_analysis",
                "carbon_prices_currency_per_tco2e": [0.0, 1000.0, 10000.0],
            },
            "cost_weight": None,
            "carbon_weight": None,
            "demand_charge_per_kw": None,
        }
    )
    replay = ReplaySettings.model_validate(replay_payload)
    config = base.model_copy(update={"replay": replay})

    result = ReplayStudyRunner(clock=lambda: pd.Timestamp("2026-08-21T00:00:00Z")).run(config)
    expected = {pareto_policy_name(value) for value in (0.0, 1000.0, 10000.0)}

    assert isinstance(replay.objective, ParetoAnalysisObjectiveConfig)
    assert expected.issubset(set(result.replay.metrics["policy"]))
    assert len(result.replay.metrics) == 5 + len(expected)
    assert result.replay.status["feasible"].all()
    run = ReplayArtifactWriter().write(result, tmp_path / "runs")
    assert verify_run(run).valid
