from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import pandas as pd

from climadc.contracts.frames import PredictionFrame
from climadc.errors import ConfigurationError
from climadc.forecasting.baselines import (
    _BaseForecaster,
    _feature_frame,
    _LINEAR_FEATURES,
    _prediction_frame,
    _prediction_record,
    _validate_declared_fields,
    _validate_prediction_request,
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
            raise ConfigurationError("Install climadc[lightgbm]") from exc
        self._regressor_type: type[Any] = module.LGBMRegressor
        self.models_: dict[str, Any] = {}
        self._feature_anchor: pd.Timestamp | None = None

    def fit(self, train: pd.DataFrame, context: Mapping[str, object]) -> LightGBMForecaster:
        selected = self._fit_training_data(train)
        self._feature_anchor = pd.Timestamp(selected["valid_time"].min())
        self.models_ = {}
        for site_id in self.sites_:
            site_rows = selected.loc[selected["site_id"] == site_id]
            if len(site_rows) < 2:
                raise ConfigurationError(
                    f"Insufficient training rows for site {site_id!r}; need at least 2"
                )
            times = [pd.Timestamp(value) for value in site_rows["valid_time"]]
            features = _feature_frame(times, self.features, self._feature_anchor)
            model = self._regressor_type(n_estimators=20, random_state=0, verbosity=-1)
            model.fit(features, site_rows["value"].to_numpy(dtype=float))
            self.models_[site_id] = model
        return self

    def predict(self, origins: pd.DatetimeIndex, horizon: pd.Timedelta) -> PredictionFrame:
        _, unit = self._require_fitted()
        checked_origins, checked_horizon = _validate_prediction_request(origins, horizon)
        if self._feature_anchor is None:
            raise ConfigurationError(f"{type(self).__name__} is not fitted")
        valid_times = checked_origins + checked_horizon
        features = _feature_frame(valid_times, self.features, self._feature_anchor)
        records: list[dict[str, object]] = []
        for site_id in self.sites_:
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
