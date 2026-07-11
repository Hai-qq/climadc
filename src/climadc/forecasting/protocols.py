from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import pandas as pd

from climadc.contracts.frames import PredictionFrame


@runtime_checkable
class Forecaster(Protocol):
    def fit(self, train: pd.DataFrame, context: Mapping[str, object]) -> Forecaster: ...

    def predict(self, origins: pd.DatetimeIndex, horizon: pd.Timedelta) -> PredictionFrame: ...
