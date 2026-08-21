from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from climadc.contracts import (
    ClimateForecastFrame,
    DCTelemetryFrame,
    FlexibleWorkloadFrame,
    GridSignalFrame,
)
from climadc.errors import ConfigurationError, ContractError
from climadc.replay import POLICY_NAMES, ReplayConfig, ReplayEngine, TemperatureSensitivePUEModel
from climadc.replay.engine import RISK_AWARE_POLICY
from climadc.replay.study import _forecast_metrics


DECISION_TIME = pd.Timestamp("2026-01-01 00:00", tz="UTC")
SLOTS = pd.date_range(DECISION_TIME, periods=4, freq="1h")


def _climate(values: list[float] | None = None) -> ClimateForecastFrame:
    temperatures = values or [20.0, 20.0, 20.0, 20.0]
    return ClimateForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "site_id": ["dc-1"] * 4,
                "issue_time": [DECISION_TIME - pd.Timedelta(hours=2)] * 4,
                "available_at": [DECISION_TIME - pd.Timedelta(hours=1)] * 4,
                "valid_time": SLOTS,
                "variable": ["air_temperature"] * 4,
                "value": temperatures,
                "unit": ["degC"] * 4,
                "source": ["forecast-fixture"] * 4,
                "quantile": [pd.NA] * 4,
                "member": [pd.NA] * 4,
            }
        )
    )


def _actual_weather(values: list[float] | None = None) -> DCTelemetryFrame:
    temperatures = values or [25.0, 25.0, 15.0, 15.0]
    return DCTelemetryFrame.from_pandas(
        pd.DataFrame(
            {
                "site_id": ["dc-1"] * 4,
                "device_id": ["weather-station"] * 4,
                "event_time": SLOTS,
                "available_at": SLOTS + pd.Timedelta(minutes=5),
                "metric": ["air_temperature"] * 4,
                "value": temperatures,
                "unit": ["degC"] * 4,
                "quality": ["observed"] * 4,
            }
        )
    )


def _grid() -> GridSignalFrame:
    forecast_price = [0.4, 0.1, 0.3, 0.2]
    forecast_carbon = [100.0, 500.0, 200.0, 400.0]
    actual_price = [0.5, 0.5, 0.1, 0.1]
    actual_carbon = [500.0, 400.0, 100.0, 200.0]
    rows: list[dict[str, object]] = []
    for signal, unit, forecast, actual in (
        ("energy_price", "GBP / kWh", forecast_price, actual_price),
        ("carbon_intensity", "gCO2e / kWh", forecast_carbon, actual_carbon),
    ):
        for slot, predicted, realized in zip(SLOTS, forecast, actual, strict=True):
            rows.append(
                {
                    "site_id": "dc-1",
                    "region_id": "GB-13",
                    "issue_time": DECISION_TIME - pd.Timedelta(hours=2),
                    "available_at": DECISION_TIME - pd.Timedelta(hours=1),
                    "valid_time": slot,
                    "signal": signal,
                    "value": predicted,
                    "unit": unit,
                    "source": "grid-fixture",
                    "quality": "forecast",
                    "quantile": pd.NA,
                }
            )
            rows.append(
                {
                    "site_id": "dc-1",
                    "region_id": "GB-13",
                    "issue_time": pd.NaT,
                    "available_at": slot + pd.Timedelta(minutes=30),
                    "valid_time": slot,
                    "signal": signal,
                    "value": realized,
                    "unit": unit,
                    "source": "grid-fixture",
                    "quality": "observed",
                    "quantile": pd.NA,
                }
            )
    return GridSignalFrame.from_pandas(pd.DataFrame(rows))


def _workload(*, energy: float = 2.0, max_power: float = 1.0) -> FlexibleWorkloadFrame:
    return FlexibleWorkloadFrame.from_pandas(
        pd.DataFrame(
            {
                "job_id": ["batch-1"],
                "site_id": ["dc-1"],
                "release_time": [DECISION_TIME],
                "available_at": [DECISION_TIME],
                "deadline": [DECISION_TIME + pd.Timedelta(hours=4)],
                "energy": [energy],
                "energy_unit": ["kWh"],
                "max_power": [max_power],
                "power_unit": ["kW"],
                "preemptible": [True],
                "priority": [1.0],
            }
        )
    )


def _config(**updates: object) -> ReplayConfig:
    values: dict[str, object] = {
        "site_id": "dc-1",
        "horizon": pd.Timedelta(hours=4),
        "interval": pd.Timedelta(hours=1),
        "it_capacity_kw": 3.0,
        "fixed_it_power_kw": 1.0,
        "cost_weight": 1.0,
        "carbon_weight": 1.0,
        "demand_charge_per_kw": 0.0,
    }
    values.update(updates)
    return ReplayConfig(**values)  # type: ignore[arg-type]


def _risk_forecasts() -> tuple[ClimateForecastFrame, GridSignalFrame]:
    climate = _climate().to_pandas()
    climate["quantile"] = climate["quantile"].astype("Float64")
    climate_quantile = climate.copy(deep=True)
    climate_quantile["quantile"] = 0.9
    climate = pd.DataFrame.from_records(
        [*climate.to_dict(orient="records"), *climate_quantile.to_dict(orient="records")],
        columns=climate.columns,
    )

    grid = _grid().to_pandas()
    grid["quantile"] = grid["quantile"].astype("Float64")
    quantiles = grid.loc[grid["quality"] == "forecast"].copy(deep=True)
    quantiles["quantile"] = 0.9
    price = quantiles["signal"] == "energy_price"
    quantiles.loc[price, "value"] = [0.05, 1.0, 1.0, 1.0]
    grid = pd.DataFrame.from_records(
        [*grid.to_dict(orient="records"), *quantiles.to_dict(orient="records")],
        columns=grid.columns,
    )
    return ClimateForecastFrame.from_pandas(climate), GridSignalFrame.from_pandas(grid)


def _run(
    *,
    climate: ClimateForecastFrame | None = None,
    weather: DCTelemetryFrame | None = None,
    grid: GridSignalFrame | None = None,
    workload: FlexibleWorkloadFrame | None = None,
    config: ReplayConfig | None = None,
):
    return ReplayEngine(TemperatureSensitivePUEModel()).run(
        decision_time=DECISION_TIME,
        climate_forecast=climate or _climate(),
        actual_weather=weather or _actual_weather(),
        grid_signals=grid or _grid(),
        workload=workload or _workload(),
        config=config or _config(),
    )


def test_replay_runs_all_policies_and_settles_unit_aware_metrics() -> None:
    result = _run()

    assert tuple(result.status["policy"]) == POLICY_NAMES
    assert result.status["feasible"].all()
    assert set(result.metrics["policy"]) == set(POLICY_NAMES)
    assert result.currency == "GBP"
    assert result.accepted_jobs == 1
    assert result.future_jobs == 0
    assert set(result.allocations["policy"]) == set(POLICY_NAMES)
    assert set(result.profiles["policy"]) == set(POLICY_NAMES)

    metrics = result.metrics.set_index("policy")
    for column in (
        "facility_energy_kwh",
        "it_energy_kwh",
        "cooling_energy_kwh",
        "estimated_location_based_emissions_kgco2e",
        "energy_cost",
        "peak_kw",
        "completed_jobs",
        "deadline_violations",
        "unserved_energy_kwh",
        "energy_balance_error_kwh",
        "shifted_energy_kwh",
        "realized_objective",
        "objective_regret",
    ):
        assert np.isfinite(metrics[column].to_numpy(dtype=float)).all()

    assert metrics.loc["asap", "shifted_energy_kwh"] == pytest.approx(0.0)
    assert metrics.loc["peak", "peak_kw"] < metrics.loc["asap", "peak_kw"]
    assert metrics.loc["oracle", "objective_regret"] == pytest.approx(0.0, abs=1e-8)
    assert (metrics["objective_regret"] >= -1e-8).all()
    assert (metrics["deadline_violations"] == 0.0).all()
    assert (metrics["unserved_energy_kwh"] <= 1e-8).all()
    assert (metrics["energy_balance_error_kwh"] <= 1e-8).all()


def test_policy_profiles_reflect_distinct_asap_price_carbon_and_oracle_choices() -> None:
    result = _run()
    flexible = result.profiles.pivot(
        index="valid_time", columns="policy", values="flexible_it_power_kw"
    )

    np.testing.assert_allclose(flexible["asap"], [1.0, 1.0, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(flexible["price"], [0.0, 1.0, 0.0, 1.0], atol=1e-8)
    np.testing.assert_allclose(flexible["carbon"], [1.0, 0.0, 1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(flexible["oracle"], [0.0, 0.0, 1.0, 1.0], atol=1e-8)


def test_risk_aware_policy_uses_complete_declared_upper_quantile_scenario() -> None:
    climate, grid = _risk_forecasts()

    result = _run(
        climate=climate,
        grid=grid,
        workload=_workload(energy=1.0),
        config=_config(cost_weight=1.0, carbon_weight=0.0, risk_quantile=0.9),
    )

    assert tuple(result.status["policy"]) == (*POLICY_NAMES[:-1], RISK_AWARE_POLICY, "oracle")
    allocations = result.allocations.loc[result.allocations["power_kw"] > 1e-8]
    joint_slot = allocations.loc[allocations["policy"] == "joint", "valid_time"].item()
    risk_slot = allocations.loc[allocations["policy"] == RISK_AWARE_POLICY, "valid_time"].item()
    assert joint_slot == SLOTS[1]
    assert risk_slot == SLOTS[0]
    risk_profile = result.profiles.loc[result.profiles["policy"] == RISK_AWARE_POLICY]
    assert set(risk_profile["decision_basis"]) == {"quantile:0.9"}
    assert risk_profile.set_index("valid_time").loc[SLOTS[0], "decision_energy_price"] == 0.05


def test_single_window_risk_diagnostics_backtest_each_marginal_quantile() -> None:
    climate, grid = _risk_forecasts()
    result = _run(
        climate=climate,
        grid=grid,
        workload=_workload(energy=1.0),
        config=_config(cost_weight=1.0, carbon_weight=0.0, risk_quantile=0.9),
    )

    forecast_metrics = _forecast_metrics(result, risk_quantile=0.9)
    diagnostics = forecast_metrics["upper_quantile_diagnostics"]

    assert forecast_metrics["upper_quantile_diagnostics_status"] == "computed"
    assert isinstance(diagnostics, dict)
    assert diagnostics["method"] == "committed_slot_marginal_backtest"
    assert diagnostics["nominal_quantile"] == pytest.approx(0.9)
    assert diagnostics["sample_count"] == 4
    signals = diagnostics["signals"]
    assert isinstance(signals, dict)

    temperature = signals["temperature"]
    assert temperature["covered_count"] == 2
    assert temperature["exceedance_count"] == 2
    assert temperature["empirical_coverage"] == pytest.approx(0.5)
    assert temperature["coverage_gap"] == pytest.approx(-0.4)
    assert temperature["mean_positive_exceedance"] == pytest.approx(2.5)
    assert temperature["mean_exceedance_when_exceeded"] == pytest.approx(5.0)
    assert temperature["maximum_exceedance"] == pytest.approx(5.0)
    assert temperature["pinball_loss"] == pytest.approx(2.5)
    assert temperature["wilson_95_lower"] == pytest.approx(0.1500389892)
    assert temperature["wilson_95_upper"] == pytest.approx(0.8499610108)

    price = signals["energy_price"]
    assert price["empirical_coverage"] == pytest.approx(0.75)
    assert price["mean_positive_exceedance"] == pytest.approx(0.1125)
    assert price["pinball_loss"] == pytest.approx(0.15875)

    carbon = signals["carbon_intensity"]
    assert carbon["unit"] == "kgCO2e/kWh"
    assert carbon["empirical_coverage"] == pytest.approx(0.75)
    assert carbon["mean_positive_exceedance"] == pytest.approx(0.1)
    assert carbon["pinball_loss"] == pytest.approx(0.1)


def test_risk_diagnostics_count_equality_as_covered_without_exceedance() -> None:
    climate, grid = _risk_forecasts()
    result = _run(
        climate=climate,
        weather=_actual_weather([20.0, 15.0, 15.0, 15.0]),
        grid=grid,
        workload=_workload(energy=1.0),
        config=_config(cost_weight=1.0, carbon_weight=0.0, risk_quantile=0.9),
    )

    forecast_metrics = _forecast_metrics(result, risk_quantile=0.9)
    diagnostics = forecast_metrics["upper_quantile_diagnostics"]
    assert isinstance(diagnostics, dict)
    signals = diagnostics["signals"]
    assert isinstance(signals, dict)
    temperature = signals["temperature"]

    assert temperature["covered_count"] == 4
    assert temperature["empirical_coverage"] == pytest.approx(1.0)
    assert temperature["exceedance_count"] == 0
    assert temperature["mean_positive_exceedance"] == pytest.approx(0.0)
    assert temperature["mean_exceedance_when_exceeded"] == pytest.approx(0.0)
    assert temperature["maximum_exceedance"] == pytest.approx(0.0)


def test_risk_aware_policy_rejects_incomplete_quantile_inputs() -> None:
    with pytest.raises(ConfigurationError, match="quantile 0.9"):
        _run(config=_config(risk_quantile=0.9))


def test_asap_runs_higher_priority_job_first_when_capacity_is_contended() -> None:
    jobs = pd.concat([_workload().to_pandas()] * 2, ignore_index=True)
    jobs["job_id"] = ["high-priority", "low-priority"]
    jobs["deadline"] = DECISION_TIME + pd.Timedelta(hours=2)
    jobs["energy"] = 1.0
    jobs["max_power"] = 1.0
    jobs["priority"] = [10.0, 0.0]

    result = _run(
        workload=FlexibleWorkloadFrame.from_pandas(jobs),
        config=_config(it_capacity_kw=2.0),
    )
    asap = result.allocations.loc[
        (result.allocations["policy"] == "asap") & (result.allocations["power_kw"] > 1e-8)
    ].sort_values("valid_time")

    assert list(asap["job_id"]) == ["high-priority", "low-priority"]
    assert list(asap["valid_time"]) == [SLOTS[0], SLOTS[1]]


def test_replay_ignores_forecasts_that_arrive_after_decision_time() -> None:
    frame = _grid().to_pandas()
    late = frame.loc[
        (frame["quality"] == "forecast")
        & (frame["signal"] == "energy_price")
        & (frame["valid_time"] == SLOTS[1])
    ].copy()
    late["issue_time"] = DECISION_TIME - pd.Timedelta(minutes=30)
    late["available_at"] = DECISION_TIME + pd.Timedelta(minutes=1)
    late["value"] = -100.0
    frame = pd.concat([frame, late], ignore_index=True)

    result = _run(grid=GridSignalFrame.from_pandas(frame))
    profile = result.profiles.loc[result.profiles["policy"] == "asap"].set_index("valid_time")

    assert profile.loc[SLOTS[1], "forecast_energy_price"] == pytest.approx(0.1)


def test_replay_uses_realized_values_only_for_settlement() -> None:
    result = _run()
    profile = result.profiles.loc[result.profiles["policy"] == "asap"].set_index("valid_time")

    assert profile.loc[SLOTS[0], "forecast_carbon_kgco2e_per_kwh"] == pytest.approx(0.1)
    assert profile.loc[SLOTS[0], "actual_carbon_kgco2e_per_kwh"] == pytest.approx(0.5)
    assert profile.loc[SLOTS[0], "forecast_pue"] != profile.loc[SLOTS[0], "actual_pue"]


def test_replay_normalizes_compatible_engineering_units() -> None:
    grid = _grid().to_pandas()
    price = grid["signal"] == "energy_price"
    grid.loc[price, "value"] = grid.loc[price, "value"] * 1000.0
    grid.loc[price, "unit"] = "GBP / MWh"
    carbon = grid["signal"] == "carbon_intensity"
    grid.loc[carbon, "unit"] = "kgCO2e / MWh"

    climate = _climate().to_pandas()
    climate["value"] = climate["value"] * 9.0 / 5.0 + 32.0
    climate["unit"] = "degF"
    weather = _actual_weather().to_pandas()
    weather["value"] = weather["value"] * 9.0 / 5.0 + 32.0
    weather["unit"] = "degF"

    result = _run(
        climate=ClimateForecastFrame.from_pandas(climate),
        weather=DCTelemetryFrame.from_pandas(weather),
        grid=GridSignalFrame.from_pandas(grid),
    )
    profile = result.profiles.loc[result.profiles["policy"] == "asap"].set_index("valid_time")

    assert profile.loc[SLOTS[0], "forecast_temperature_c"] == pytest.approx(20.0)
    assert profile.loc[SLOTS[0], "actual_temperature_c"] == pytest.approx(25.0)
    assert profile.loc[SLOTS[0], "forecast_energy_price"] == pytest.approx(0.4)
    assert profile.loc[SLOTS[0], "forecast_carbon_kgco2e_per_kwh"] == pytest.approx(0.1)


def test_demand_charge_is_settled_and_changes_price_policy_peak() -> None:
    without_charge = _run()
    with_charge = _run(config=_config(demand_charge_per_kw=10.0))
    baseline = without_charge.metrics.set_index("policy")
    charged = with_charge.metrics.set_index("policy")

    assert charged.loc["price", "peak_kw"] < baseline.loc["price", "peak_kw"]
    assert charged.loc["price", "demand_charge"] == pytest.approx(
        10.0 * charged.loc["price", "peak_kw"]
    )
    assert charged.loc["price", "energy_cost"] == pytest.approx(
        charged.loc["price", "energy_charge"] + charged.loc["price", "demand_charge"]
    )


def test_replay_excludes_future_unknown_jobs_without_silently_dropping_arrived_jobs() -> None:
    jobs = _workload().to_pandas()
    future = jobs.copy()
    future["job_id"] = "future-job"
    future["release_time"] = DECISION_TIME + pd.Timedelta(minutes=30)
    future["available_at"] = DECISION_TIME + pd.Timedelta(hours=1)
    workload = FlexibleWorkloadFrame.from_pandas(pd.concat([jobs, future], ignore_index=True))

    result = _run(workload=workload)

    assert result.accepted_jobs == 1
    assert result.future_jobs == 1
    assert set(result.allocations["job_id"]) == {"batch-1"}


def test_replay_handles_a_window_with_only_future_jobs() -> None:
    future = _workload().to_pandas()
    future["release_time"] = DECISION_TIME + pd.Timedelta(minutes=30)
    future["available_at"] = DECISION_TIME + pd.Timedelta(hours=1)

    result = _run(workload=FlexibleWorkloadFrame.from_pandas(future))

    assert result.status["feasible"].all()
    assert result.accepted_jobs == 0
    assert result.future_jobs == 1
    assert result.allocations.empty
    assert (result.metrics["completed_jobs"] == 0.0).all()


def test_ambiguous_latest_forecast_is_rejected() -> None:
    frame = _grid().to_pandas()
    duplicate = frame.loc[
        (frame["quality"] == "forecast")
        & (frame["signal"] == "energy_price")
        & (frame["valid_time"] == SLOTS[0])
    ].copy()
    duplicate["source"] = "equally-fresh-provider"
    frame = pd.concat([frame, duplicate], ignore_index=True)

    with pytest.raises(ConfigurationError, match="ambiguous latest forecast"):
        _run(grid=GridSignalFrame.from_pandas(frame))


def test_capacity_infeasibility_is_reported_for_every_policy() -> None:
    result = _run(
        workload=_workload(energy=4.0, max_power=4.0),
        config=_config(it_capacity_kw=1.5),
    )

    assert not result.status["feasible"].any()
    assert result.metrics.empty
    assert result.allocations.empty
    assert result.profiles.empty
    assert set(result.violations) == set(POLICY_NAMES)
    assert all(result.violations[policy] for policy in POLICY_NAMES)


def test_job_deadline_beyond_horizon_is_reported_as_infeasible() -> None:
    frame = _workload().to_pandas()
    frame.loc[0, "deadline"] = DECISION_TIME + pd.Timedelta(hours=5)
    workload = FlexibleWorkloadFrame.from_pandas(frame)

    result = _run(workload=workload)

    assert not result.status["feasible"].any()
    assert all("horizon" in " ".join(items) for items in result.violations.values())


def test_missing_required_signal_slot_fails_before_optimization() -> None:
    frame = _grid().to_pandas()
    missing = ~(
        (frame["quality"] == "observed")
        & (frame["signal"] == "energy_price")
        & (frame["valid_time"] == SLOTS[-1])
    )

    with pytest.raises(ConfigurationError, match="energy_price.*realized.*slots"):
        _run(grid=GridSignalFrame.from_pandas(frame.loc[missing]))


def test_mixed_price_currencies_are_rejected() -> None:
    frame = _grid().to_pandas()
    usd = frame["signal"] == "energy_price"
    frame.loc[usd & (frame["valid_time"] == SLOTS[-1]), "unit"] = "USD / kWh"

    with pytest.raises((ContractError, ConfigurationError), match="unit|currency"):
        _run(grid=GridSignalFrame.from_pandas(frame))


@pytest.mark.parametrize(
    "decision_time",
    [pd.Timestamp("2026-01-01 00:00"), pd.Timestamp("2026-01-01 08:00", tz="Asia/Shanghai")],
)
def test_replay_requires_exact_utc_decision_time(decision_time: pd.Timestamp) -> None:
    with pytest.raises(ConfigurationError, match="decision_time.*UTC"):
        ReplayEngine(TemperatureSensitivePUEModel()).run(
            decision_time=decision_time,
            climate_forecast=_climate(),
            actual_weather=_actual_weather(),
            grid_signals=_grid(),
            workload=_workload(),
            config=_config(),
        )


def test_replay_result_defensively_copies_frames_and_is_frozen() -> None:
    result = _run()
    original = float(result.metrics.loc[0, "energy_cost"])

    mutated = result.metrics
    mutated.loc[0, "energy_cost"] = 999.0

    assert result.to_metrics().loc[0, "energy_cost"] == pytest.approx(original)
    with pytest.raises(FrozenInstanceError):
        result.currency = "USD"  # type: ignore[misc]


def test_replay_rejects_wrong_input_types() -> None:
    engine = ReplayEngine(TemperatureSensitivePUEModel())

    with pytest.raises(ContractError, match="climate_forecast"):
        engine.run(
            decision_time=DECISION_TIME,
            climate_forecast=pd.DataFrame(),  # type: ignore[arg-type]
            actual_weather=_actual_weather(),
            grid_signals=_grid(),
            workload=_workload(),
            config=_config(),
        )


def test_replay_rejects_invalid_facility_model_output() -> None:
    class InvalidPUEModel:
        def pue(self, ambient_temperature_c: np.ndarray) -> np.ndarray:
            return np.full_like(ambient_temperature_c, 0.9)

    with pytest.raises(ContractError, match="PUE.*>= 1"):
        ReplayEngine(InvalidPUEModel()).run(
            decision_time=DECISION_TIME,
            climate_forecast=_climate(),
            actual_weather=_actual_weather(),
            grid_signals=_grid(),
            workload=_workload(),
            config=_config(),
        )
