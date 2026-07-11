from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from climadc.contracts.frames import ClimateForecastFrame, DCTelemetryFrame
from climadc.validation.leakage import LeakageGuard

_FORECAST_GROUP = ("site_id", "valid_time", "variable", "quantile", "member")


def _deep_copy_object_cells(frame: pd.DataFrame) -> pd.DataFrame:
    copied: pd.DataFrame = frame.copy(deep=True)
    for position, dtype in enumerate(frame.dtypes):
        if pd.api.types.is_object_dtype(dtype):
            values = [deepcopy(value) for value in frame.iloc[:, position].tolist()]
            copied.isetitem(position, pd.Series(values, index=copied.index, dtype=object))
    return copied


def _require_origin(origin: object) -> pd.Timestamp:
    if (
        not isinstance(origin, pd.Timestamp)
        or origin.tzinfo is None
        or origin.utcoffset() is None
        or str(origin.tzinfo) != "UTC"
        or origin.utcoffset() != pd.Timedelta(0)
    ):
        raise ValueError("origin must be a scalar timezone-aware UTC pandas Timestamp")
    return origin


def _require_horizon(horizon: object) -> pd.Timedelta:
    if not isinstance(horizon, pd.Timedelta) or pd.isna(horizon) or horizon <= pd.Timedelta(0):
        raise ValueError("horizon must be a positive pandas Timedelta")
    return horizon


@dataclass(frozen=True)
class DecisionView:
    origin: pd.Timestamp
    target_time: pd.Timestamp
    forecast: pd.DataFrame
    telemetry_history: pd.DataFrame
    observed_targets: pd.DataFrame


class DecisionViewBuilder:
    def build(
        self,
        climate: ClimateForecastFrame,
        telemetry: DCTelemetryFrame,
        origin: pd.Timestamp,
        horizon: pd.Timedelta,
    ) -> DecisionView:
        checked_origin = _require_origin(origin)
        checked_horizon = _require_horizon(horizon)
        target_time = checked_origin + checked_horizon
        guard = LeakageGuard()

        legal_climate, _ = guard.safe_subset(climate.to_pandas(), checked_origin)
        forecast = legal_climate.loc[
            (legal_climate["valid_time"] > checked_origin)
            & (legal_climate["valid_time"] <= target_time)
        ].copy(deep=True)
        forecast.sort_values(
            by=[*_FORECAST_GROUP, "issue_time", "available_at", "source"],
            kind="mergesort",
            na_position="last",
            inplace=True,
        )
        forecast.drop_duplicates(subset=list(_FORECAST_GROUP), keep="last", inplace=True)
        forecast.reset_index(drop=True, inplace=True)

        telemetry_data = telemetry.to_pandas()
        legal_telemetry, _ = guard.safe_subset(telemetry_data, checked_origin)
        telemetry_history = legal_telemetry.loc[
            legal_telemetry["event_time"] <= checked_origin
        ].copy(deep=True)
        telemetry_history.reset_index(drop=True, inplace=True)

        observed_targets = _deep_copy_object_cells(
            telemetry_data.loc[
                (telemetry_data["event_time"] == target_time)
                & (telemetry_data["quality"] == "observed")
            ]
        )
        observed_targets.reset_index(drop=True, inplace=True)

        return DecisionView(
            origin=checked_origin,
            target_time=target_time,
            forecast=forecast,
            telemetry_history=telemetry_history,
            observed_targets=observed_targets,
        )
