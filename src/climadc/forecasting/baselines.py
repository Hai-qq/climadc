from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, cast

import pandas as pd
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from climadc.contracts.frames import PREDICTION_COLUMNS, PredictionFrame
from climadc.errors import ConfigurationError

_TRAINING_COLUMNS = ("site_id", "valid_time", "available_at", "target", "value", "unit")
_CALENDAR_FIELDS = frozenset({"hour", "dayofweek", "month", "dayofyear"})
_GROUP_FIELDS = _CALENDAR_FIELDS | {"site_id"}
_LINEAR_FEATURES = _CALENDAR_FIELDS | {"elapsed_hours"}


def _require_nonempty_label(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value


def _normalize_timestamp_column(frame: pd.DataFrame, column: str) -> None:
    normalized: list[pd.Timestamp] = []
    for value in frame[column].tolist():
        if not pd.api.types.is_scalar(value) or pd.isna(value):
            raise ConfigurationError(f"{column} must contain timezone-aware timestamps")
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(f"{column} must contain timezone-aware timestamps") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ConfigurationError(f"{column} must contain timezone-aware timestamps")
        normalized.append(timestamp.tz_convert("UTC"))
    frame[column] = pd.Series(normalized, index=frame.index, dtype="datetime64[ns, UTC]")


def _validate_training_frame(train: pd.DataFrame, target: str) -> pd.DataFrame:
    if not isinstance(train, pd.DataFrame):
        raise ConfigurationError("train must be a pandas DataFrame")
    missing = sorted(set(_TRAINING_COLUMNS).difference(train.columns))
    if missing:
        raise ConfigurationError(f"Training data missing required columns: {missing}")

    checked = train.copy(deep=True)
    for column in ("valid_time", "available_at"):
        _normalize_timestamp_column(checked, column)

    for column in ("site_id", "target", "unit"):
        invalid = checked[column].map(lambda value: not isinstance(value, str) or not value.strip())
        if bool(invalid.any()):
            raise ConfigurationError(f"{column} must contain non-empty strings")

    invalid_values = checked["value"].map(
        lambda value: (
            not isinstance(value, Real) or isinstance(value, bool) or not isfinite(float(value))
        )
    )
    if bool(invalid_values.any()):
        raise ConfigurationError("value must contain finite real numbers")

    selected = cast(pd.DataFrame, checked.loc[checked["target"] == target].copy(deep=True))
    if selected.empty:
        raise ConfigurationError(f"No training rows for target {target!r}")
    units = selected["unit"].unique().tolist()
    if len(units) != 1:
        raise ConfigurationError("Selected target must have exactly one unit label")

    selected["_input_order"] = range(len(selected))
    selected.reset_index(drop=True, inplace=True)
    return selected


def _validate_origins(origins: object) -> pd.DatetimeIndex:
    if not isinstance(origins, pd.DatetimeIndex):
        raise ConfigurationError("origins must be a pandas DatetimeIndex")
    if origins.empty:
        raise ConfigurationError("origins must be non-empty")
    if origins.hasnans:
        raise ConfigurationError("origins must not contain NaT")
    if (
        origins.tz is None
        or str(origins.tz) != "UTC"
        or any(timestamp.utcoffset() != pd.Timedelta(0) for timestamp in origins)
    ):
        raise ConfigurationError("origins must contain timezone-aware exact UTC timestamps")
    if not origins.is_unique:
        raise ConfigurationError("origins must contain unique timestamps")
    if not origins.is_monotonic_increasing:
        raise ConfigurationError("origins must be sorted in increasing order")
    return cast(pd.DatetimeIndex, origins.copy())


def _validate_horizon(horizon: object) -> pd.Timedelta:
    if not isinstance(horizon, pd.Timedelta) or pd.isna(horizon) or horizon <= pd.Timedelta(0):
        raise ConfigurationError("horizon must be a positive pandas Timedelta")
    return horizon


def _validate_prediction_request(
    origins: object, horizon: object
) -> tuple[pd.DatetimeIndex, pd.Timedelta]:
    return _validate_origins(origins), _validate_horizon(horizon)


def _prediction_record(
    *,
    site_id: str,
    origin: pd.Timestamp,
    horizon: pd.Timedelta,
    target: str,
    value: float,
    unit: str,
    model_id: str,
) -> dict[str, object]:
    return {
        "site_id": site_id,
        "issue_time": origin,
        "valid_time": origin + horizon,
        "target": target,
        "value": float(value),
        "unit": unit,
        "model_id": model_id,
        "quantile": pd.NA,
    }


def _prediction_frame(records: list[dict[str, object]]) -> PredictionFrame:
    frame = pd.DataFrame.from_records(records, columns=PREDICTION_COLUMNS)
    return PredictionFrame.from_pandas(frame)


def _timestamp_field(timestamp: pd.Timestamp, field: str) -> int:
    if field == "hour":
        return timestamp.hour
    if field == "dayofweek":
        return timestamp.dayofweek
    if field == "month":
        return timestamp.month
    if field == "dayofyear":
        return timestamp.dayofyear
    raise ConfigurationError(f"Unknown calendar field {field!r}")


def _validate_declared_fields(
    name: str,
    fields: Sequence[str],
    allowed: frozenset[str],
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    declared = tuple(fields)
    if not allow_empty and not declared:
        raise ConfigurationError(f"{name} must declare at least one feature")
    if len(set(declared)) != len(declared):
        raise ConfigurationError(f"{name} must not contain duplicates")
    unknown = sorted(set(declared).difference(allowed))
    if unknown:
        raise ConfigurationError(f"{name} contains unsupported fields: {unknown}")
    return declared


def _feature_frame(
    times: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    features: tuple[str, ...],
    anchor: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for timestamp in times:
        row: dict[str, float] = {}
        for feature in features:
            if feature == "elapsed_hours":
                row[feature] = float((timestamp - anchor) / pd.Timedelta("1h"))
            else:
                row[feature] = float(_timestamp_field(timestamp, feature))
        rows.append(row)
    return cast(
        pd.DataFrame,
        pd.DataFrame.from_records(rows, columns=list(features)).astype(float),
    )


@dataclass(frozen=True)
class _PreparedTrainingData:
    frame: pd.DataFrame
    unit: str
    sites: tuple[str, ...]


class _BaseForecaster:
    def __init__(self, *, target: str, model_id: str) -> None:
        self.target = _require_nonempty_label("target", target)
        self.model_id = _require_nonempty_label("model_id", model_id)
        self.unit: str | None = None
        self.sites_: tuple[str, ...] = ()
        self._train: pd.DataFrame | None = None

    def _prepare_training_data(self, train: pd.DataFrame) -> _PreparedTrainingData:
        selected = _validate_training_frame(train, self.target)
        return _PreparedTrainingData(
            frame=selected,
            unit=str(selected["unit"].iloc[0]),
            sites=tuple(sorted(str(site) for site in selected["site_id"].unique())),
        )

    def _commit_training_data(self, prepared: _PreparedTrainingData) -> None:
        self._train = prepared.frame
        self.unit = prepared.unit
        self.sites_ = prepared.sites

    def _require_fitted(self) -> tuple[pd.DataFrame, str]:
        if self._train is None or self.unit is None:
            raise ConfigurationError(f"{type(self).__name__} is not fitted")
        return self._train, self.unit


class PersistenceForecaster(_BaseForecaster):
    def __init__(self, *, target: str, model_id: str = "persistence") -> None:
        super().__init__(target=target, model_id=model_id)

    def fit(self, train: pd.DataFrame, context: Mapping[str, object]) -> PersistenceForecaster:
        prepared = self._prepare_training_data(train)
        self._commit_training_data(prepared)
        return self

    def predict(self, origins: pd.DatetimeIndex, horizon: pd.Timedelta) -> PredictionFrame:
        train, unit = self._require_fitted()
        checked_origins, checked_horizon = _validate_prediction_request(origins, horizon)
        records: list[dict[str, object]] = []
        for site_id in self.sites_:
            site_rows = train.loc[train["site_id"] == site_id]
            for origin in checked_origins:
                legal = site_rows.loc[
                    (site_rows["valid_time"] <= origin) & (site_rows["available_at"] <= origin)
                ].sort_values(["valid_time", "available_at", "_input_order"], kind="mergesort")
                if legal.empty:
                    raise ConfigurationError(
                        f"No legal history for site {site_id!r} at origin {origin}"
                    )
                records.append(
                    _prediction_record(
                        site_id=site_id,
                        origin=origin,
                        horizon=checked_horizon,
                        target=self.target,
                        value=float(legal.iloc[-1]["value"]),
                        unit=unit,
                        model_id=self.model_id,
                    )
                )
        return _prediction_frame(records)


class SeasonalNaiveForecaster(_BaseForecaster):
    def __init__(
        self,
        *,
        target: str,
        period: pd.Timedelta,
        model_id: str = "seasonal_naive",
    ) -> None:
        super().__init__(target=target, model_id=model_id)
        if not isinstance(period, pd.Timedelta) or pd.isna(period) or period <= pd.Timedelta(0):
            raise ConfigurationError("period must be a positive pandas Timedelta")
        self.period = period

    def fit(self, train: pd.DataFrame, context: Mapping[str, object]) -> SeasonalNaiveForecaster:
        prepared = self._prepare_training_data(train)
        self._commit_training_data(prepared)
        return self

    def predict(self, origins: pd.DatetimeIndex, horizon: pd.Timedelta) -> PredictionFrame:
        train, unit = self._require_fitted()
        checked_origins, checked_horizon = _validate_prediction_request(origins, horizon)
        records: list[dict[str, object]] = []
        for site_id in self.sites_:
            site_rows = train.loc[train["site_id"] == site_id]
            for origin in checked_origins:
                reference_time = origin + checked_horizon - self.period
                legal = site_rows.loc[
                    (site_rows["valid_time"] == reference_time)
                    & (site_rows["available_at"] <= origin)
                ].sort_values(["available_at", "_input_order"], kind="mergesort")
                if legal.empty:
                    raise ConfigurationError(
                        f"No exact seasonal reference for site {site_id!r} at origin {origin}"
                    )
                records.append(
                    _prediction_record(
                        site_id=site_id,
                        origin=origin,
                        horizon=checked_horizon,
                        target=self.target,
                        value=float(legal.iloc[-1]["value"]),
                        unit=unit,
                        model_id=self.model_id,
                    )
                )
        return _prediction_frame(records)


class ClimatologyForecaster(_BaseForecaster):
    def __init__(
        self,
        *,
        target: str,
        group_by: tuple[str, ...],
        model_id: str = "climatology",
    ) -> None:
        super().__init__(target=target, model_id=model_id)
        self.group_by = _validate_declared_fields(
            "group_by", group_by, _GROUP_FIELDS, allow_empty=True
        )
        self.group_means_: dict[tuple[object, ...], float] = {}

    def _group_key(self, site_id: str, valid_time: pd.Timestamp) -> tuple[object, ...]:
        return tuple(
            site_id if field == "site_id" else _timestamp_field(valid_time, field)
            for field in self.group_by
        )

    def fit(self, train: pd.DataFrame, context: Mapping[str, object]) -> ClimatologyForecaster:
        prepared = self._prepare_training_data(train)
        values: dict[tuple[object, ...], list[float]] = {}
        for row in prepared.frame.itertuples(index=False):
            key = self._group_key(str(row.site_id), pd.Timestamp(cast(Any, row.valid_time)))
            values.setdefault(key, []).append(float(cast(Any, row.value)))
        group_means = {
            key: float(sum(group_values) / len(group_values))
            for key, group_values in values.items()
        }
        self._commit_training_data(prepared)
        self.group_means_ = group_means
        return self

    def predict(self, origins: pd.DatetimeIndex, horizon: pd.Timedelta) -> PredictionFrame:
        _, unit = self._require_fitted()
        checked_origins, checked_horizon = _validate_prediction_request(origins, horizon)
        records: list[dict[str, object]] = []
        for site_id in self.sites_:
            for origin in checked_origins:
                key = self._group_key(site_id, origin + checked_horizon)
                if key not in self.group_means_:
                    raise ConfigurationError(f"No climatology group {key!r} for site {site_id!r}")
                records.append(
                    _prediction_record(
                        site_id=site_id,
                        origin=origin,
                        horizon=checked_horizon,
                        target=self.target,
                        value=self.group_means_[key],
                        unit=unit,
                        model_id=self.model_id,
                    )
                )
        return _prediction_frame(records)


class LinearForecaster(_BaseForecaster):
    def __init__(
        self,
        *,
        target: str,
        features: tuple[str, ...],
        model_id: str = "linear",
    ) -> None:
        super().__init__(target=target, model_id=model_id)
        self.features = _validate_declared_fields(
            "features", features, _LINEAR_FEATURES, allow_empty=False
        )
        self.models_: dict[str, Pipeline] = {}
        self.feature_anchors_: dict[str, pd.Timestamp] = {}

    def fit(self, train: pd.DataFrame, context: Mapping[str, object]) -> LinearForecaster:
        prepared = self._prepare_training_data(train)
        site_frames = {
            site_id: cast(
                pd.DataFrame,
                prepared.frame.loc[prepared.frame["site_id"] == site_id],
            )
            for site_id in prepared.sites
        }
        for site_id, site_rows in site_frames.items():
            if len(site_rows) < 2:
                raise ConfigurationError(
                    f"Insufficient training rows for site {site_id!r}; need at least 2"
                )
        feature_anchors = {
            site_id: pd.Timestamp(site_rows["valid_time"].min())
            for site_id, site_rows in site_frames.items()
        }
        models: dict[str, Pipeline] = {}
        for site_id, site_rows in site_frames.items():
            times = [pd.Timestamp(value) for value in site_rows["valid_time"]]
            features = _feature_frame(times, self.features, feature_anchors[site_id])
            pipeline = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
            pipeline.fit(features, site_rows["value"].to_numpy(dtype=float))
            models[site_id] = pipeline
        self._commit_training_data(prepared)
        self.feature_anchors_ = feature_anchors
        self.models_ = models
        return self

    def predict(self, origins: pd.DatetimeIndex, horizon: pd.Timedelta) -> PredictionFrame:
        _, unit = self._require_fitted()
        checked_origins, checked_horizon = _validate_prediction_request(origins, horizon)
        valid_times = checked_origins + checked_horizon
        records: list[dict[str, object]] = []
        for site_id in self.sites_:
            features = _feature_frame(valid_times, self.features, self.feature_anchors_[site_id])
            predictions = self.models_[site_id].predict(features)
            for origin, value in zip(checked_origins, predictions, strict=True):
                records.append(
                    _prediction_record(
                        site_id=site_id,
                        origin=origin,
                        horizon=checked_horizon,
                        target=self.target,
                        value=float(value),
                        unit=unit,
                        model_id=self.model_id,
                    )
                )
        return _prediction_frame(records)
