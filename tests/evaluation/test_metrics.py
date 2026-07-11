import math

import numpy as np
import pytest

from climadc.errors import ConfigurationError
from climadc.evaluation import point_metrics, probabilistic_metrics


def test_point_metrics_match_reference_formulas() -> None:
    metrics = point_metrics(actual=[1.0, 2.0, 4.0], predicted=[1.0, 3.0, 2.0])

    assert set(metrics) == {"mae", "rmse", "wape", "r2"}
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(math.sqrt(5.0 / 3.0))
    assert metrics["wape"] == pytest.approx(3.0 / 7.0)
    assert metrics["r2"] == pytest.approx(-0.0714285714285714)


def test_point_metrics_constant_actuals_use_force_finite_r2() -> None:
    assert point_metrics([2.0, 2.0], [2.0, 2.0])["r2"] == 1.0
    assert point_metrics([2.0, 2.0], [2.0, 3.0])["r2"] == 0.0


def test_point_metrics_require_two_samples_for_r2() -> None:
    with pytest.raises(ConfigurationError, match="at least 2"):
        point_metrics([1.0], [1.0])


def test_point_metrics_reject_zero_wape_denominator() -> None:
    with pytest.raises(ConfigurationError, match="WAPE.*zero"):
        point_metrics([0.0, 0.0], [0.0, 1.0])


def test_probabilistic_metrics_report_pinball_coverage_and_width() -> None:
    metrics = probabilistic_metrics(
        actual=[1.0, 3.0],
        lower=[0.0, 2.0],
        median=[0.5, 3.5],
        upper=[1.0, 4.0],
        alpha=0.2,
    )

    assert set(metrics) == {"pinball_loss", "coverage", "mean_width"}
    assert metrics["coverage"] == 1.0
    assert metrics["mean_width"] == 1.5
    assert metrics["pinball_loss"] == pytest.approx(0.13333333333333333)


def test_probabilistic_coverage_is_inclusive() -> None:
    metrics = probabilistic_metrics(
        actual=[1.0, 3.0],
        lower=[1.0, 2.0],
        median=[1.5, 2.5],
        upper=[2.0, 3.0],
        alpha=0.2,
    )

    assert metrics["coverage"] == 1.0


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1, True])
def test_probabilistic_metrics_reject_invalid_alpha(alpha: object) -> None:
    with pytest.raises(ConfigurationError, match="alpha"):
        probabilistic_metrics([1.0], [0.0], [1.0], [2.0], alpha=alpha)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [
        ([], []),
        ([1.0, 2.0], [1.0]),
        ([1.0, float("nan")], [1.0, 2.0]),
        ([1.0, float("inf")], [1.0, 2.0]),
        ([1.0, True], [1.0, 2.0]),
        ([[1.0], [2.0]], [1.0, 2.0]),
        (np.array([[1.0, 2.0]]), [1.0, 2.0]),
    ],
)
def test_metrics_reject_bad_arrays(actual: object, predicted: object) -> None:
    with pytest.raises(ConfigurationError):
        point_metrics(actual, predicted)  # type: ignore[arg-type]


def test_probabilistic_metrics_validate_every_array_and_length() -> None:
    with pytest.raises(ConfigurationError):
        probabilistic_metrics([1.0, 2.0], [0.0, 1.0], [1.0], [2.0, 3.0], alpha=0.2)
    with pytest.raises(ConfigurationError):
        probabilistic_metrics([1.0], [0.0], [True], [2.0], alpha=0.2)


@pytest.mark.parametrize(
    ("lower", "median", "upper"),
    [
        ([2.0], [1.0], [3.0]),
        ([0.0], [3.0], [2.0]),
    ],
)
def test_probabilistic_metrics_reject_unordered_intervals(
    lower: list[float], median: list[float], upper: list[float]
) -> None:
    with pytest.raises(ConfigurationError, match="lower <= median <= upper"):
        probabilistic_metrics([1.0], lower, median, upper, alpha=0.2)
