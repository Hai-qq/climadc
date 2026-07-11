from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import cast

import numpy as np
import pandas as pd

from climadc.contracts.frames import PREDICTION_COLUMNS, PredictionFrame
from climadc.errors import ConfigurationError, LeakageError

_BASE_KEY = ("site_id", "issue_time", "valid_time", "target", "unit", "model_id")


def _validate_alpha(alpha: object) -> float:
    if (
        not isinstance(alpha, Real)
        or isinstance(alpha, bool)
        or not math.isfinite(float(alpha))
        or not 0.0 < float(alpha) < 1.0
    ):
        raise ConfigurationError("alpha must be a finite number strictly inside (0, 1)")
    return float(alpha)


@dataclass(frozen=True)
class _PredictionGroup:
    metadata: dict[str, object]
    lower: float
    median: float
    upper: float
    layout: str


def _prediction_groups(predictions: PredictionFrame, alpha: float) -> list[_PredictionGroup]:
    frame = predictions.to_pandas()
    groups: list[_PredictionGroup] = []
    layouts: set[str] = set()
    expected = (alpha / 2.0, 0.5, 1.0 - alpha / 2.0)

    for _, group in frame.groupby(list(_BASE_KEY), sort=True, dropna=False, observed=True):
        quantiles = group["quantile"]
        if len(group) == 1 and (pd.isna(quantiles.iloc[0]) or quantiles.iloc[0] == 0.5):
            value = float(group["value"].iloc[0])
            lower, median, upper = value, value, value
            layout = "point"
        elif len(group) == 3 and tuple(float(value) for value in quantiles) == expected:
            lower = float(group["value"].iloc[0])
            median = float(group["value"].iloc[1])
            upper = float(group["value"].iloc[2])
            layout = "interval"
        else:
            raise ConfigurationError(
                "Prediction quantile layout must be one point row (null or 0.5) or exactly "
                f"{expected!r} per base group"
            )

        raw_lower, raw_upper = sorted((lower, upper))
        first = group.iloc[0]
        metadata = {column: first[column] for column in _BASE_KEY}
        groups.append(
            _PredictionGroup(
                metadata=metadata,
                lower=raw_lower,
                median=median,
                upper=raw_upper,
                layout=layout,
            )
        )
        layouts.add(layout)

    if not groups:
        raise ConfigurationError("PredictionFrame must contain at least one prediction group")
    if len(layouts) != 1:
        raise ConfigurationError("PredictionFrame must not contain mixed point/interval groups")
    return groups


def _actual_array(actuals: pd.Series, expected_length: int) -> np.ndarray:
    if not isinstance(actuals, pd.Series):
        raise ConfigurationError("actuals must be a pandas Series")
    if len(actuals) != expected_length:
        raise ConfigurationError("actuals length must exactly match prediction groups")
    if not actuals.index.equals(pd.RangeIndex(expected_length)):
        raise ConfigurationError("actuals index must exactly align with prediction group order")

    values: list[float] = []
    for value in actuals.tolist():
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ConfigurationError("actuals must contain finite real numbers")
        values.append(float(value))
    return np.asarray(values, dtype=float)


def _finite_sample_adjustment(scores: np.ndarray, alpha: float) -> float:
    level = min(1.0, math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    return float(np.quantile(scores, level, method="higher"))


class SplitConformalCalibrator:
    """Split conformal calibration with explicit calibration/test time separation."""

    def __init__(self, alpha: float) -> None:
        self.alpha = _validate_alpha(alpha)
        self.adjustment_: float | None = None
        self.calibration_valid_times_: frozenset[pd.Timestamp] | None = None

    def fit(
        self,
        calibration_predictions: PredictionFrame,
        actuals: pd.Series,
    ) -> SplitConformalCalibrator:
        groups = _prediction_groups(calibration_predictions, self.alpha)
        observed = _actual_array(actuals, len(groups))
        if groups[0].layout == "point":
            scores = np.abs(np.asarray([group.median for group in groups]) - observed)
        else:
            lower = np.asarray([group.lower for group in groups])
            upper = np.asarray([group.upper for group in groups])
            scores = np.maximum(lower - observed, observed - upper)

        adjustment = _finite_sample_adjustment(scores, self.alpha)
        valid_times = frozenset(
            cast(pd.Timestamp, group.metadata["valid_time"]) for group in groups
        )
        self.adjustment_ = adjustment
        self.calibration_valid_times_ = valid_times
        return self

    def transform(self, predictions: PredictionFrame) -> PredictionFrame:
        if self.adjustment_ is None or self.calibration_valid_times_ is None:
            raise ConfigurationError("SplitConformalCalibrator is not fitted")
        groups = _prediction_groups(predictions, self.alpha)
        prediction_times = {cast(pd.Timestamp, group.metadata["valid_time"]) for group in groups}
        if prediction_times.intersection(self.calibration_valid_times_):
            raise LeakageError("Prediction valid_time overlap with calibration block")

        quantiles = (self.alpha / 2.0, 0.5, 1.0 - self.alpha / 2.0)
        records: list[dict[str, object]] = []
        for group in groups:
            values = (
                group.lower - self.adjustment_,
                group.median,
                group.upper + self.adjustment_,
            )
            for quantile, value in zip(quantiles, values, strict=True):
                records.append(
                    {
                        "site_id": group.metadata["site_id"],
                        "issue_time": group.metadata["issue_time"],
                        "valid_time": group.metadata["valid_time"],
                        "target": group.metadata["target"],
                        "value": value,
                        "unit": group.metadata["unit"],
                        "model_id": group.metadata["model_id"],
                        "quantile": quantile,
                    }
                )
        frame = pd.DataFrame.from_records(records, columns=PREDICTION_COLUMNS)
        return PredictionFrame.from_pandas(frame)
