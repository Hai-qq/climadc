from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from climadc.adapters.local import (
    read_climate,
    read_flexible_workload,
    read_grid_signals,
    read_telemetry,
)
from climadc.contracts import (
    ClimateForecastFrame,
    DCTelemetryFrame,
    FlexibleWorkloadFrame,
    GridSignalFrame,
)
from climadc.errors import ConfigurationError, ContractError
from climadc.replay.config import ReplayStudyConfig
from climadc.replay.engine import RISK_AWARE_POLICY, ReplayEngine, ReplayResult
from climadc.replay.manifest import SourceManifest
from climadc.replay.rolling import RollingReplayEngine, RollingReplayResult

Clock = Callable[[], pd.Timestamp]
_QUANTILE_CONFIDENCE_LEVEL = 0.95


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _exact_utc(value: object, field: str) -> pd.Timestamp:
    if (
        not isinstance(value, pd.Timestamp)
        or pd.isna(value)
        or value.tzinfo is None
        or value.utcoffset() is None
        or str(value.tzinfo) != "UTC"
    ):
        raise ConfigurationError(f"{field} must be a scalar pandas Timestamp in exact UTC")
    return value


def assumptions_payload(config: ReplayStudyConfig) -> dict[str, object]:
    """Return the portable, path-independent replay assumptions."""

    return {
        "schema_version": config.schema_version,
        "study_id": config.study_id,
        "decision_time": config.decision_time.isoformat(),
        "replay": config.replay.model_dump(mode="json"),
        "rolling": None if config.rolling is None else config.rolling.model_dump(mode="json"),
        "facility_model": config.facility_model.model_dump(mode="json"),
        "assumptions": config.assumptions,
        "limitations": config.limitations,
    }


@dataclass(frozen=True)
class ReplayStudyResult:
    config: ReplayStudyConfig
    manifest: SourceManifest
    climate_forecast: ClimateForecastFrame
    actual_weather: DCTelemetryFrame
    grid_signals: GridSignalFrame
    workload: FlexibleWorkloadFrame
    replay: ReplayResult | RollingReplayResult
    forecast_metrics: dict[str, object]
    input_hashes: dict[str, str]
    config_sha256: str
    started_at: pd.Timestamp


def _wilson_interval(successes: int, sample_count: int) -> tuple[float, float]:
    if sample_count <= 0 or successes < 0 or successes > sample_count:
        raise ContractError("Wilson interval requires 0 <= successes <= sample_count")
    z = NormalDist().inv_cdf(0.5 + _QUANTILE_CONFIDENCE_LEVEL / 2.0)
    proportion = successes / sample_count
    denominator = 1.0 + z**2 / sample_count
    center = (proportion + z**2 / (2.0 * sample_count)) / denominator
    margin = (
        z
        * sqrt((proportion * (1.0 - proportion) + z**2 / (4.0 * sample_count)) / sample_count)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _quantile_signal_diagnostics(
    *,
    predicted: np.ndarray,
    actual: np.ndarray,
    quantile: float,
    unit: str,
) -> dict[str, object]:
    if predicted.shape != actual.shape or predicted.ndim != 1 or len(predicted) == 0:
        raise ContractError("quantile diagnostics require aligned non-empty one-dimensional values")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ContractError("quantile diagnostics require finite forecast and actual values")
    covered = actual <= predicted
    covered_count = int(np.count_nonzero(covered))
    sample_count = len(predicted)
    exceedance = np.maximum(actual - predicted, 0.0)
    exceeded = exceedance > 0.0
    exceedance_count = int(np.count_nonzero(exceeded))
    pinball = np.where(
        actual >= predicted,
        quantile * (actual - predicted),
        (1.0 - quantile) * (predicted - actual),
    )
    lower, upper = _wilson_interval(covered_count, sample_count)
    empirical_coverage = covered_count / sample_count
    return {
        "unit": unit,
        "sample_count": sample_count,
        "covered_count": covered_count,
        "exceedance_count": exceedance_count,
        "empirical_coverage": empirical_coverage,
        "coverage_gap": empirical_coverage - quantile,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
        "mean_positive_exceedance": float(np.mean(exceedance)),
        "mean_exceedance_when_exceeded": (
            float(np.mean(exceedance[exceeded])) if exceedance_count else 0.0
        ),
        "maximum_exceedance": float(np.max(exceedance)),
        "pinball_loss": float(np.mean(pinball)),
    }


def _upper_quantile_diagnostics(
    profiles: pd.DataFrame,
    *,
    quantile: float,
    currency: str,
) -> dict[str, object]:
    keys = ["valid_time"]
    if "decision_time" in profiles.columns:
        keys.insert(0, "decision_time")
    asap = profiles.loc[profiles["policy"] == "asap"].sort_values(keys)
    risk = profiles.loc[profiles["policy"] == RISK_AWARE_POLICY].sort_values(keys)
    if risk.empty:
        raise ContractError("configured risk replay is missing the risk-aware settlement profile")
    if risk.duplicated(keys).any():
        raise ContractError("risk-aware settlement profile contains duplicate committed slots")
    if list(risk[keys].itertuples(index=False, name=None)) != list(
        asap[keys].itertuples(index=False, name=None)
    ):
        raise ContractError("risk-aware and ASAP settlement slots differ")
    expected_basis = f"quantile:{quantile:g}"
    if set(risk["decision_basis"]) != {expected_basis}:
        raise ContractError("risk-aware settlement profile has an unexpected decision basis")

    signals = {
        "temperature": (
            "decision_temperature_c",
            "actual_temperature_c",
            "degC",
        ),
        "energy_price": (
            "decision_energy_price",
            "actual_energy_price",
            f"{currency}/kWh",
        ),
        "carbon_intensity": (
            "decision_carbon_kgco2e_per_kwh",
            "actual_carbon_kgco2e_per_kwh",
            "kgCO2e/kWh",
        ),
    }
    return {
        "method": "committed_slot_marginal_backtest",
        "nominal_quantile": quantile,
        "confidence_level": _QUANTILE_CONFIDENCE_LEVEL,
        "sample_count": len(risk),
        "signals": {
            name: _quantile_signal_diagnostics(
                predicted=risk[predicted_column].to_numpy(dtype=float),
                actual=risk[actual_column].to_numpy(dtype=float),
                quantile=quantile,
                unit=unit,
            )
            for name, (predicted_column, actual_column, unit) in signals.items()
        },
    }


def _forecast_metrics(
    result: ReplayResult | RollingReplayResult,
    *,
    risk_quantile: float | None,
) -> dict[str, object]:
    profiles = result.profiles
    interval_status = (
        "point_forecasts_only" if risk_quantile is None else "two_sided_interval_not_available"
    )
    if profiles.empty:
        return {
            "status": "not_computed",
            "reason": "replay was infeasible before settlement profiles were produced",
            "interval_coverage": None,
            "interval_coverage_status": interval_status,
            "upper_quantile_diagnostics": None,
            "upper_quantile_diagnostics_status": (
                "not_configured" if risk_quantile is None else "not_computed_infeasible"
            ),
        }
    sort_columns = ["valid_time"]
    if "decision_time" in profiles.columns:
        sort_columns.insert(0, "decision_time")
    profile = profiles.loc[profiles["policy"] == "asap"].sort_values(sort_columns)
    temperature_error = np.abs(
        profile["forecast_temperature_c"].to_numpy(dtype=float)
        - profile["actual_temperature_c"].to_numpy(dtype=float)
    )
    price_error = np.abs(
        profile["forecast_energy_price"].to_numpy(dtype=float)
        - profile["actual_energy_price"].to_numpy(dtype=float)
    )
    carbon_error = np.abs(
        profile["forecast_carbon_kgco2e_per_kwh"].to_numpy(dtype=float)
        - profile["actual_carbon_kgco2e_per_kwh"].to_numpy(dtype=float)
    )
    upper_quantile_diagnostics = (
        None
        if risk_quantile is None
        else _upper_quantile_diagnostics(
            profiles,
            quantile=risk_quantile,
            currency=result.currency,
        )
    )
    return {
        "status": "computed",
        "temperature_mae_c": float(np.mean(temperature_error)),
        "energy_price_mae_per_kwh": float(np.mean(price_error)),
        "carbon_intensity_mae_gco2e_per_kwh": float(np.mean(carbon_error) * 1000.0),
        "interval_coverage": None,
        "interval_coverage_status": interval_status,
        "upper_quantile_diagnostics": upper_quantile_diagnostics,
        "upper_quantile_diagnostics_status": (
            "not_configured" if risk_quantile is None else "computed"
        ),
    }


class ReplayStudyRunner:
    """Load verified local inputs, execute the replay kernel, and freeze one result."""

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock = clock if clock is not None else (lambda: pd.Timestamp.now(tz="UTC"))

    @staticmethod
    def _read_inputs(
        config: ReplayStudyConfig,
    ) -> tuple[
        ClimateForecastFrame,
        DCTelemetryFrame,
        GridSignalFrame,
        FlexibleWorkloadFrame,
    ]:
        climate = read_climate(
            config.inputs.climate_forecast.path,
            config.inputs.climate_forecast.format,
            {},
            config.inputs.climate_forecast.timezone,
        )
        weather = read_telemetry(
            config.inputs.actual_weather.path,
            config.inputs.actual_weather.format,
            {},
            config.inputs.actual_weather.timezone,
        )
        grid = read_grid_signals(
            config.inputs.grid_signals.path,
            config.inputs.grid_signals.format,
            {},
            config.inputs.grid_signals.timezone,
        )
        workload = read_flexible_workload(
            config.inputs.workload.path,
            config.inputs.workload.format,
            {},
            config.inputs.workload.timezone,
        )
        return climate, weather, grid, workload

    def run(self, config: ReplayStudyConfig) -> ReplayStudyResult:
        manifest = SourceManifest.from_yaml(config.source_manifest)
        if manifest.study_id != config.study_id:
            raise ConfigurationError("Replay config and source manifest study_id values differ")
        required = {
            item.path.resolve()
            for item in (
                config.inputs.climate_forecast,
                config.inputs.actual_weather,
                config.inputs.grid_signals,
                config.inputs.workload,
            )
        }
        input_hashes = manifest.validate_files(config.source_manifest.parent, required)
        climate, weather, grid, workload = self._read_inputs(config)
        replay_config = config.replay.to_replay_config()
        if config.rolling is None:
            replay: ReplayResult | RollingReplayResult = ReplayEngine(
                config.facility_model.build()
            ).run(
                decision_time=config.decision_timestamp,
                climate_forecast=climate,
                actual_weather=weather,
                grid_signals=grid,
                workload=workload,
                config=replay_config,
            )
        else:
            replay = RollingReplayEngine(config.facility_model.build()).run(
                start_time=config.decision_timestamp,
                periods=config.rolling.periods,
                step=config.rolling.step_timedelta,
                climate_forecast=climate,
                actual_weather=weather,
                grid_signals=grid,
                workload=workload,
                config=replay_config,
            )
        payload: dict[str, Any] = {
            "assumptions": assumptions_payload(config),
            "manifest": manifest.model_dump(mode="json"),
        }
        config_sha256 = hashlib.sha256(_canonical_json(payload)).hexdigest()
        started_at = _exact_utc(self._clock(), "replay clock")
        return ReplayStudyResult(
            config=config,
            manifest=manifest,
            climate_forecast=climate,
            actual_weather=weather,
            grid_signals=grid,
            workload=workload,
            replay=replay,
            forecast_metrics=_forecast_metrics(
                replay,
                risk_quantile=config.replay.risk_quantile,
            ),
            input_hashes=input_hashes,
            config_sha256=config_sha256,
            started_at=started_at,
        )
