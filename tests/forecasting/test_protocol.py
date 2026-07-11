from collections.abc import Mapping

import pandas as pd

from climadc.contracts.frames import PredictionFrame
from climadc.forecasting import Forecaster


class UserForecaster:
    def fit(self, train: pd.DataFrame, context: Mapping[str, object]) -> "UserForecaster":
        return self

    def predict(self, origins: pd.DatetimeIndex, horizon: pd.Timedelta) -> PredictionFrame:
        raise NotImplementedError


class MissingPredict:
    def fit(self, train: pd.DataFrame, context: Mapping[str, object]) -> "MissingPredict":
        return self


def test_forecaster_protocol_accepts_structural_user_model() -> None:
    assert isinstance(UserForecaster(), Forecaster)


def test_forecaster_protocol_rejects_model_missing_required_method() -> None:
    assert not isinstance(MissingPredict(), Forecaster)
