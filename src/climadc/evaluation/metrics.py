from __future__ import annotations

import math
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from climadc.errors import ConfigurationError


def _finite_vector(values: ArrayLike, name: str) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a one-dimensional numeric array") from exc
    if raw.ndim != 1 or len(raw) == 0:
        raise ConfigurationError(f"{name} must be a non-empty one-dimensional array")
    converted: list[float] = []
    for value in raw.tolist():
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ConfigurationError(f"{name} must contain finite real numbers")
        converted.append(float(value))
    return np.asarray(converted, dtype=float)


def _aligned_vectors(**arrays: ArrayLike) -> dict[str, np.ndarray]:
    checked = {name: _finite_vector(values, name) for name, values in arrays.items()}
    lengths = {len(values) for values in checked.values()}
    if len(lengths) != 1:
        raise ConfigurationError("metric arrays must have exactly equal lengths")
    return checked


def _alpha(alpha: object) -> float:
    if (
        not isinstance(alpha, Real)
        or isinstance(alpha, bool)
        or not math.isfinite(float(alpha))
        or not 0.0 < float(alpha) < 1.0
    ):
        raise ConfigurationError("alpha must be a finite number strictly inside (0, 1)")
    return float(alpha)


def point_metrics(actual: ArrayLike, predicted: ArrayLike) -> dict[str, float]:
    """Compute deterministic point metrics on aligned finite vectors."""

    arrays = _aligned_vectors(actual=actual, predicted=predicted)
    y = arrays["actual"]
    prediction = arrays["predicted"]
    if len(y) < 2:
        raise ConfigurationError("point metrics require at least 2 samples for finite R2")
    absolute_error = np.abs(y - prediction)
    denominator = float(np.sum(np.abs(y)))
    if denominator == 0.0:
        raise ConfigurationError("WAPE denominator is zero")

    residual_sum = float(np.sum(np.square(y - prediction)))
    total_sum = float(np.sum(np.square(y - np.mean(y))))
    if total_sum == 0.0:
        r2 = 1.0 if residual_sum == 0.0 else 0.0
    else:
        r2 = 1.0 - residual_sum / total_sum
    return {
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(np.square(y - prediction)))),
        "wape": float(np.sum(absolute_error) / denominator),
        "r2": float(r2),
    }


def _pinball(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    residual = actual - predicted
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def probabilistic_metrics(
    actual: ArrayLike,
    lower: ArrayLike,
    median: ArrayLike,
    upper: ArrayLike,
    alpha: float,
) -> dict[str, float]:
    """Compute pinball loss, inclusive interval coverage, and mean width."""

    checked_alpha = _alpha(alpha)
    arrays = _aligned_vectors(actual=actual, lower=lower, median=median, upper=upper)
    y = arrays["actual"]
    lo = arrays["lower"]
    med = arrays["median"]
    hi = arrays["upper"]
    if bool(np.any((lo > med) | (med > hi))):
        raise ConfigurationError("probabilistic intervals require lower <= median <= upper")

    quantiles = (checked_alpha / 2.0, 0.5, 1.0 - checked_alpha / 2.0)
    losses = (
        _pinball(y, lo, quantiles[0]),
        _pinball(y, med, quantiles[1]),
        _pinball(y, hi, quantiles[2]),
    )
    return {
        "pinball_loss": float(np.mean(losses)),
        "coverage": float(np.mean((lo <= y) & (y <= hi))),
        "mean_width": float(np.mean(hi - lo)),
    }
