from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import pandas as pd

from climadc.contracts.frames import PredictionFrame
from climadc.errors import ConfigurationError
from climadc.forecasting.baselines import (
    _BaseForecaster,
    _LINEAR_FEATURES,
    _feature_frame,
    _prediction_frame,
    _prediction_record,
    _validate_declared_fields,
    _validate_prediction_request,
)

_NATIVE_LIBRARY_MARKERS = (
    "libomp",
    "libgomp",
    "vcomp",
    ".dylib",
    ".so",
    ".dll",
    "dll load failed",
    "shared object",
)
_MISSING_LOADER_MARKERS = (
    "cannot open",
    "dll load failed",
    "image not found",
    "library not loaded",
    "no such file",
    "not found",
)


def _is_lightgbm_unavailable(error: BaseException | None) -> bool:
    if isinstance(error, ModuleNotFoundError):
        return error.name == "lightgbm"
    if not isinstance(error, (ImportError, OSError)):
        return False
    message = str(error).lower()
    return any(marker in message for marker in _NATIVE_LIBRARY_MARKERS) and any(
        marker in message for marker in _MISSING_LOADER_MARKERS
    )


class LightGBMForecaster(_BaseForecaster):
    def __init__(
        self,
        *,
        target: str,
        features: tuple[str, ...],
        model_id: str = "lightgbm",
    ) -> None:
        super().__init__(target=target, model_id=model_id)
        self.features = _validate_declared_fields(
            "features", features, _LINEAR_FEATURES, allow_empty=False
        )
        try:
            module = importlib.import_module("lightgbm")
        except (ImportError, OSError) as exc:
            if _is_lightgbm_unavailable(exc):
                raise ConfigurationError("Install climadc[lightgbm]") from exc
            raise
        self._regressor_type: type[Any] = module.LGBMRegressor
        self.models_: dict[str, Any] = {}
        self.feature_anchors_: dict[str, pd.Timestamp] = {}

    def fit(self, train: pd.DataFrame, context: Mapping[str, object]) -> LightGBMForecaster:
        prepared = self._prepare_training_data(train)
        site_frames = {
            site_id: prepared.frame.loc[prepared.frame["site_id"] == site_id]
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
        models: dict[str, Any] = {}
        for site_id, site_rows in site_frames.items():
            times = [pd.Timestamp(value) for value in site_rows["valid_time"]]
            features = _feature_frame(times, self.features, feature_anchors[site_id])
            model = self._regressor_type(n_estimators=20, random_state=0, verbosity=-1)
            model.fit(features, site_rows["value"].to_numpy(dtype=float))
            models[site_id] = model
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
