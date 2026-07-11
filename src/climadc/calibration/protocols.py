from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from climadc.contracts.frames import PredictionFrame


@runtime_checkable
class Calibrator(Protocol):
    """Structural interface for post-hoc prediction calibrators."""

    def fit(
        self,
        calibration_predictions: PredictionFrame,
        actuals: pd.Series,
    ) -> Calibrator: ...

    def transform(self, predictions: PredictionFrame) -> PredictionFrame: ...
