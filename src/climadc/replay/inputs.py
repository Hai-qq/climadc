from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from pint.errors import PintError

from climadc.contracts import (
    ClimateForecastFrame,
    DCTelemetryFrame,
    FlexibleWorkloadFrame,
    GridSignalFrame,
)
from climadc.errors import ConfigurationError, ContractError
from climadc.replay.models import ReplayConfig
from climadc.validation.units import UNIT_REGISTRY

_CURRENCIES = ("GBP", "USD", "EUR", "CNY")
_REALIZED_QUALITY_ORDER = {"imputed": 0, "estimated": 1, "observed": 2}
_GRID_REALIZED_QUALITY_ORDER = {"estimated": 0, "observed": 1}


@dataclass(frozen=True)
class PreparedReplayInputs:
    decision_time: pd.Timestamp
    slots: pd.DatetimeIndex
    interval_hours: float
    forecast_temperature_c: np.ndarray
    risk_temperature_c: np.ndarray | None
    actual_temperature_c: np.ndarray
    forecast_price_per_kwh: np.ndarray
    risk_price_per_kwh: np.ndarray | None
    actual_price_per_kwh: np.ndarray
    forecast_carbon_kgco2e_per_kwh: np.ndarray
    risk_carbon_kgco2e_per_kwh: np.ndarray | None
    actual_carbon_kgco2e_per_kwh: np.ndarray
    currency: str
    jobs: pd.DataFrame
    eligible: np.ndarray
    accepted_jobs: int
    future_jobs: int


def _require_exact_utc_decision_time(value: object) -> pd.Timestamp:
    if (
        not isinstance(value, pd.Timestamp)
        or value.tzinfo is None
        or value.utcoffset() is None
        or str(value.tzinfo) != "UTC"
        or value.utcoffset() != pd.Timedelta(0)
    ):
        raise ConfigurationError("decision_time must be an exact UTC pandas Timestamp")
    return value


def _require_type(value: object, expected: type[object], name: str) -> None:
    if not isinstance(value, expected):
        raise ContractError(f"{name} must be a {expected.__name__}")


def _missing_slots(frame: pd.DataFrame, time_column: str, slots: pd.DatetimeIndex) -> list[str]:
    available = set(frame[time_column].tolist())
    return [slot.isoformat() for slot in slots if slot not in available]


def _point_representation(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    point = frame.loc[frame["quantile"].isna()].copy()
    if point.empty:
        point = frame.loc[frame["quantile"] == 0.5].copy()
    if point.empty:
        raise ConfigurationError(f"{label} requires a point or median forecast for every slot")
    return cast(pd.DataFrame, point)


def _choose_latest_forecast(
    frame: pd.DataFrame,
    *,
    slots: pd.DatetimeIndex,
    time_column: str,
    decision_time: pd.Timestamp,
    label: str,
    prefer_null_member: bool = False,
    forecast_quantile: float | None = None,
) -> pd.DataFrame:
    legal = frame.loc[
        frame[time_column].isin(slots)
        & (frame["available_at"] <= decision_time)
        & (frame["issue_time"] <= decision_time)
    ].copy()
    selected: list[pd.Series] = []
    for slot in slots:
        slot_rows = legal.loc[legal[time_column] == slot]
        if slot_rows.empty:
            raise ConfigurationError(
                f"{label} forecast is missing required slots: {[slot.isoformat()]}"
            )
        if forecast_quantile is None:
            candidates = _point_representation(slot_rows, label=f"{label} at {slot.isoformat()}")
        else:
            quantiles = pd.to_numeric(slot_rows["quantile"], errors="coerce").to_numpy(dtype=float)
            candidates = slot_rows.loc[
                np.isclose(quantiles, forecast_quantile, rtol=0.0, atol=1e-12)
            ].copy()
            if candidates.empty:
                raise ConfigurationError(
                    f"{label} requires quantile {forecast_quantile:g} at {slot.isoformat()}"
                )
        if prefer_null_member and "member" in candidates and candidates["member"].isna().any():
            candidates = candidates.loc[candidates["member"].isna()].copy()
        latest_issue = candidates["issue_time"].max()
        candidates = candidates.loc[candidates["issue_time"] == latest_issue]
        latest_available = candidates["available_at"].max()
        candidates = candidates.loc[candidates["available_at"] == latest_available]
        if len(candidates) != 1:
            raise ConfigurationError(
                f"{label} has an ambiguous latest forecast at {slot.isoformat()}"
            )
        selected.append(candidates.iloc[0])
    result: pd.DataFrame = pd.DataFrame(selected).reset_index(drop=True)
    return result


def _choose_realized(
    frame: pd.DataFrame,
    *,
    slots: pd.DatetimeIndex,
    time_column: str,
    label: str,
    quality_order: dict[str, int],
) -> pd.DataFrame:
    candidates = frame.loc[frame[time_column].isin(slots)].copy()
    selected: list[pd.Series] = []
    for slot in slots:
        rows = candidates.loc[candidates[time_column] == slot].copy()
        if rows.empty:
            continue
        rows["_quality_rank"] = rows["quality"].map(quality_order).fillna(-1)
        best_quality = rows["_quality_rank"].max()
        rows = rows.loc[rows["_quality_rank"] == best_quality]
        latest_available = rows["available_at"].max()
        rows = rows.loc[rows["available_at"] == latest_available]
        if len(rows) != 1:
            raise ConfigurationError(f"{label} has ambiguous realized data at {slot.isoformat()}")
        selected.append(rows.iloc[0].drop(labels=["_quality_rank"]))
    if len(selected) != len(slots):
        missing = _missing_slots(candidates, time_column, slots)
        raise ConfigurationError(f"{label} realized data is missing required slots: {missing}")
    result: pd.DataFrame = pd.DataFrame(selected).reset_index(drop=True)
    return result


def _convert_values(frame: pd.DataFrame, target_unit: str, *, label: str) -> np.ndarray:
    values: list[float] = []
    for value, unit in zip(frame["value"], frame["unit"], strict=True):
        try:
            converted = UNIT_REGISTRY.Quantity(float(value), str(unit)).to(target_unit)
        except (PintError, TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(f"{label} cannot be converted to {target_unit}") from exc
        values.append(float(converted.magnitude))
    result = np.asarray(values, dtype=float)
    if len(result) != len(frame) or not np.isfinite(result).all():
        raise ConfigurationError(f"{label} conversion produced invalid values")
    return result


def _currency_for_units(units: pd.Series) -> str:
    currencies: set[str] = set()
    for label in units.tolist():
        try:
            unit = UNIT_REGISTRY.parse_units(str(label))
        except (PintError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid energy price unit: {label!r}") from exc
        matches = [
            currency
            for currency in _CURRENCIES
            if unit.is_compatible_with(UNIT_REGISTRY.parse_units(f"{currency} / kWh"))
        ]
        if len(matches) != 1:
            raise ConfigurationError(f"energy price unit has unsupported currency: {label!r}")
        currencies.add(matches[0])
    if len(currencies) != 1:
        raise ConfigurationError("energy price currency must be consistent across replay inputs")
    return next(iter(currencies))


def _prepare_climate(
    climate_forecast: ClimateForecastFrame,
    actual_weather: DCTelemetryFrame,
    *,
    config: ReplayConfig,
    decision_time: pd.Timestamp,
    slots: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    forecast = climate_forecast.to_pandas()
    forecast = forecast.loc[
        (forecast["site_id"] == config.site_id)
        & (forecast["variable"] == config.temperature_variable)
    ]
    selected_forecast = _choose_latest_forecast(
        forecast,
        slots=slots,
        time_column="valid_time",
        decision_time=decision_time,
        label=config.temperature_variable,
        prefer_null_member=True,
    )
    selected_risk = None
    if config.risk_quantile is not None:
        selected_risk = _choose_latest_forecast(
            forecast,
            slots=slots,
            time_column="valid_time",
            decision_time=decision_time,
            label=config.temperature_variable,
            prefer_null_member=True,
            forecast_quantile=config.risk_quantile,
        )

    actual = actual_weather.to_pandas()
    actual = actual.loc[
        (actual["site_id"] == config.site_id) & (actual["metric"] == config.weather_metric)
    ]
    selected_actual = _choose_realized(
        actual,
        slots=slots,
        time_column="event_time",
        label=config.weather_metric,
        quality_order=_REALIZED_QUALITY_ORDER,
    )
    return (
        _convert_values(selected_forecast, "degC", label="forecast temperature"),
        (
            None
            if selected_risk is None
            else _convert_values(selected_risk, "degC", label="risk temperature")
        ),
        _convert_values(selected_actual, "degC", label="actual temperature"),
    )


def _prepare_grid(
    grid_signals: GridSignalFrame,
    *,
    config: ReplayConfig,
    decision_time: pd.Timestamp,
    slots: pd.DatetimeIndex,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    str,
]:
    grid = grid_signals.to_pandas()
    grid = grid.loc[grid["site_id"] == config.site_id]
    selected: dict[tuple[str, str], pd.DataFrame] = {}
    for signal in ("energy_price", "carbon_intensity"):
        signal_rows = grid.loc[grid["signal"] == signal]
        selected[(signal, "forecast")] = _choose_latest_forecast(
            signal_rows.loc[signal_rows["quality"] == "forecast"],
            slots=slots,
            time_column="valid_time",
            decision_time=decision_time,
            label=signal,
        )
        if config.risk_quantile is not None:
            selected[(signal, "risk")] = _choose_latest_forecast(
                signal_rows.loc[signal_rows["quality"] == "forecast"],
                slots=slots,
                time_column="valid_time",
                decision_time=decision_time,
                label=signal,
                forecast_quantile=config.risk_quantile,
            )
        selected[(signal, "realized")] = _choose_realized(
            signal_rows.loc[signal_rows["quality"] != "forecast"],
            slots=slots,
            time_column="valid_time",
            label=signal,
            quality_order=_GRID_REALIZED_QUALITY_ORDER,
        )

    price_units = pd.concat(
        [
            selected[("energy_price", "forecast")]["unit"],
            selected[("energy_price", "realized")]["unit"],
        ],
        ignore_index=True,
    )
    currency = _currency_for_units(price_units)
    target_price_unit = f"{currency} / kWh"
    return (
        _convert_values(
            selected[("energy_price", "forecast")],
            target_price_unit,
            label="forecast energy_price",
        ),
        (
            None
            if config.risk_quantile is None
            else _convert_values(
                selected[("energy_price", "risk")],
                target_price_unit,
                label="risk energy_price",
            )
        ),
        _convert_values(
            selected[("energy_price", "realized")],
            target_price_unit,
            label="realized energy_price",
        ),
        _convert_values(
            selected[("carbon_intensity", "forecast")],
            "kgCO2e / kWh",
            label="forecast carbon_intensity",
        ),
        (
            None
            if config.risk_quantile is None
            else _convert_values(
                selected[("carbon_intensity", "risk")],
                "kgCO2e / kWh",
                label="risk carbon_intensity",
            )
        ),
        _convert_values(
            selected[("carbon_intensity", "realized")],
            "kgCO2e / kWh",
            label="realized carbon_intensity",
        ),
        currency,
    )


def _convert_job_value(value: object, unit: object, target: str, *, label: str) -> float:
    try:
        converted = UNIT_REGISTRY.Quantity(float(cast(float, value)), str(unit)).to(target)
    except (PintError, TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{label} cannot be converted to {target}") from exc
    result = float(converted.magnitude)
    if not np.isfinite(result):
        raise ConfigurationError(f"{label} conversion produced a non-finite value")
    return result


def _prepare_jobs(
    workload: FlexibleWorkloadFrame,
    *,
    config: ReplayConfig,
    decision_time: pd.Timestamp,
    slots: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    all_jobs = workload.to_pandas()
    all_jobs = all_jobs.loc[all_jobs["site_id"] == config.site_id].copy()
    future_jobs = int((all_jobs["available_at"] > decision_time).sum())
    jobs = all_jobs.loc[all_jobs["available_at"] <= decision_time].copy()
    jobs.sort_values(["deadline", "release_time", "job_id"], kind="mergesort", inplace=True)
    jobs.reset_index(drop=True, inplace=True)
    jobs["energy_kwh"] = [
        _convert_job_value(value, unit, "kWh", label=f"job {job_id} energy")
        for job_id, value, unit in zip(
            jobs["job_id"], jobs["energy"], jobs["energy_unit"], strict=True
        )
    ]
    jobs["max_power_kw"] = [
        _convert_job_value(value, unit, "kW", label=f"job {job_id} max_power")
        for job_id, value, unit in zip(
            jobs["job_id"], jobs["max_power"], jobs["power_unit"], strict=True
        )
    ]

    slot_ends = slots + config.interval
    eligible = np.zeros((len(jobs), len(slots)), dtype=bool)
    for job_index, row in jobs.iterrows():
        earliest = max(cast(pd.Timestamp, row["release_time"]), decision_time)
        eligible[job_index, :] = np.asarray(
            (slots >= earliest) & (slot_ends <= cast(pd.Timestamp, row["deadline"])),
            dtype=bool,
        )
    return jobs, eligible, future_jobs


def prepare_replay_inputs(
    *,
    decision_time: object,
    climate_forecast: object,
    actual_weather: object,
    grid_signals: object,
    workload: object,
    config: object,
) -> PreparedReplayInputs:
    checked_decision_time = _require_exact_utc_decision_time(decision_time)
    _require_type(climate_forecast, ClimateForecastFrame, "climate_forecast")
    _require_type(actual_weather, DCTelemetryFrame, "actual_weather")
    _require_type(grid_signals, GridSignalFrame, "grid_signals")
    _require_type(workload, FlexibleWorkloadFrame, "workload")
    _require_type(config, ReplayConfig, "config")
    checked_config = cast(ReplayConfig, config)
    slots = pd.date_range(
        checked_decision_time,
        periods=checked_config.slot_count,
        freq=checked_config.interval,
    )
    forecast_temperature, risk_temperature, actual_temperature = _prepare_climate(
        cast(ClimateForecastFrame, climate_forecast),
        cast(DCTelemetryFrame, actual_weather),
        config=checked_config,
        decision_time=checked_decision_time,
        slots=slots,
    )
    (
        forecast_price,
        risk_price,
        actual_price,
        forecast_carbon,
        risk_carbon,
        actual_carbon,
        currency,
    ) = _prepare_grid(
        cast(GridSignalFrame, grid_signals),
        config=checked_config,
        decision_time=checked_decision_time,
        slots=slots,
    )
    jobs, eligible, future_jobs = _prepare_jobs(
        cast(FlexibleWorkloadFrame, workload),
        config=checked_config,
        decision_time=checked_decision_time,
        slots=slots,
    )
    interval_hours = float(checked_config.interval / pd.Timedelta(hours=1))
    return PreparedReplayInputs(
        decision_time=checked_decision_time,
        slots=slots,
        interval_hours=interval_hours,
        forecast_temperature_c=forecast_temperature,
        risk_temperature_c=risk_temperature,
        actual_temperature_c=actual_temperature,
        forecast_price_per_kwh=forecast_price,
        risk_price_per_kwh=risk_price,
        actual_price_per_kwh=actual_price,
        forecast_carbon_kgco2e_per_kwh=forecast_carbon,
        risk_carbon_kgco2e_per_kwh=risk_carbon,
        actual_carbon_kgco2e_per_kwh=actual_carbon,
        currency=currency,
        jobs=jobs,
        eligible=eligible,
        accepted_jobs=len(jobs),
        future_jobs=future_jobs,
    )
