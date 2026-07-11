from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from climadc.calibration import Calibrator, SplitConformalCalibrator
from climadc.contracts.frames import PREDICTION_COLUMNS, PredictionFrame
from climadc.errors import ConfigurationError, LeakageError


def _predictions(
    values: Sequence[float],
    *,
    quantiles: Sequence[float | None] | None = None,
    start: str = "2026-01-01 00:00Z",
    model_id: str = "model-a",
) -> PredictionFrame:
    if quantiles is None:
        quantiles = [None] * len(values)
    if len(values) != len(quantiles):
        raise ValueError("fixture values and quantiles must align")
    issue_time = pd.Timestamp(start)
    if len(quantiles) == 3 and set(quantiles) == {0.1, 0.5, 0.9}:
        valid_times = [issue_time + pd.Timedelta("1h")] * 3
    elif len(quantiles) == 6 and set(quantiles) == {0.1, 0.5, 0.9}:
        valid_times = [issue_time + pd.Timedelta("1h")] * 3 + [issue_time + pd.Timedelta("2h")] * 3
    elif len(quantiles) == 4 and quantiles[0] is None and set(quantiles[1:]) == {0.1, 0.5, 0.9}:
        valid_times = [issue_time + pd.Timedelta("1h")] + [issue_time + pd.Timedelta("2h")] * 3
    else:
        valid_times = [issue_time + pd.Timedelta(hours=index + 1) for index in range(len(values))]
    frame = pd.DataFrame(
        {
            "site_id": ["dc-1"] * len(values),
            "issue_time": [issue_time] * len(values),
            "valid_time": valid_times,
            "target": ["power"] * len(values),
            "value": list(values),
            "unit": ["kW"] * len(values),
            "model_id": [model_id] * len(values),
            "quantile": list(quantiles),
        },
        columns=PREDICTION_COLUMNS,
    )
    return PredictionFrame.from_pandas(frame)


class _UserCalibrator:
    def fit(self, calibration_predictions: PredictionFrame, actuals: pd.Series) -> _UserCalibrator:
        return self

    def transform(self, predictions: PredictionFrame) -> PredictionFrame:
        return predictions


class _MissingTransform:
    def fit(
        self, calibration_predictions: PredictionFrame, actuals: pd.Series
    ) -> _MissingTransform:
        return self


def test_calibrator_protocol_is_runtime_checkable_and_exact() -> None:
    assert isinstance(_UserCalibrator(), Calibrator)
    assert not isinstance(_MissingTransform(), Calibrator)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1, True])
def test_split_conformal_rejects_invalid_alpha(alpha: object) -> None:
    with pytest.raises(ConfigurationError, match="alpha"):
        SplitConformalCalibrator(alpha=alpha)  # type: ignore[arg-type]


def test_point_calibration_uses_finite_sample_higher_quantile() -> None:
    calibration = _predictions([0.0, 0.0, 0.0, 0.0])
    calibrator = SplitConformalCalibrator(alpha=0.4).fit(
        calibration, pd.Series([0.0, 1.0, 2.0, 3.0])
    )
    test = _predictions([10.0], start="2026-02-01 00:00Z")

    result = calibrator.transform(test).to_pandas()

    assert calibrator.adjustment_ == 3.0
    assert result["quantile"].tolist() == [0.2, 0.5, 0.8]
    assert result["value"].tolist() == [7.0, 10.0, 13.0]


def test_point_quantile_half_is_accepted_and_emits_three_rows() -> None:
    calibrator = SplitConformalCalibrator(alpha=0.2).fit(
        _predictions([10.0, 12.0], quantiles=[0.5, 0.5]),
        pd.Series([11.0, 10.0]),
    )

    result = calibrator.transform(
        _predictions([20.0], quantiles=[0.5], start="2026-02-01 00:00Z")
    ).to_pandas()

    assert result["quantile"].tolist() == [0.1, 0.5, 0.9]
    assert result["value"].tolist() == [18.0, 20.0, 22.0]


def test_interval_calibration_sorts_inverted_endpoints_and_preserves_median() -> None:
    calibration = _predictions(
        [12.0, 10.0, 8.0, 18.0, 20.0, 22.0],
        quantiles=[0.1, 0.5, 0.9, 0.1, 0.5, 0.9],
    )
    calibrator = SplitConformalCalibrator(alpha=0.2).fit(calibration, pd.Series([9.0, 24.0]))
    test = _predictions(
        [15.0, 14.0, 11.0],
        quantiles=[0.1, 0.5, 0.9],
        start="2026-02-01 00:00Z",
        model_id="interval-model",
    )

    result = calibrator.transform(test).to_pandas()

    assert calibrator.adjustment_ == 2.0
    assert result["value"].tolist() == [9.0, 14.0, 17.0]
    assert result["model_id"].tolist() == ["interval-model"] * 3
    assert result["unit"].tolist() == ["kW"] * 3


def test_fit_and_transform_do_not_mutate_caller_inputs() -> None:
    calibration = _predictions([9.0, 10.0, 11.0])
    actuals = pd.Series([8.0, 10.0, 12.0])
    test = _predictions([10.0], start="2026-02-01 00:00Z")
    calibration_before = calibration.to_pandas()
    actuals_before = actuals.copy(deep=True)
    test_before = test.to_pandas()

    result = SplitConformalCalibrator(alpha=0.2).fit(calibration, actuals).transform(test)
    result_frame = result.to_pandas(copy=False)
    result_frame.loc[0, "value"] = -999.0

    assert_frame_equal(calibration.to_pandas(), calibration_before)
    pd.testing.assert_series_equal(actuals, actuals_before)
    assert_frame_equal(test.to_pandas(), test_before)


def test_transform_rejects_any_calibration_valid_time_overlap() -> None:
    calibration = _predictions([9.0, 10.0, 11.0])
    calibrator = SplitConformalCalibrator(alpha=0.2).fit(calibration, pd.Series([8.0, 10.0, 12.0]))
    overlapping = _predictions([20.0], start="2026-01-01 02:00Z")

    with pytest.raises(LeakageError, match="valid_time overlap"):
        calibrator.transform(overlapping)


def test_transform_before_fit_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="not fitted"):
        SplitConformalCalibrator(alpha=0.2).transform(_predictions([1.0]))


@pytest.mark.parametrize(
    ("quantiles", "match"),
    [
        ([0.1, 0.5], "quantile layout"),
        ([0.1, 0.5, 0.8], "quantile layout"),
        ([None, 0.1, 0.5, 0.9], "mixed point/interval"),
    ],
)
def test_fit_rejects_incompatible_quantile_layout(
    quantiles: list[float | None], match: str
) -> None:
    predictions = _predictions([1.0] * len(quantiles), quantiles=quantiles)

    with pytest.raises(ConfigurationError, match=match):
        SplitConformalCalibrator(alpha=0.2).fit(predictions, pd.Series([1.0] * len(quantiles)))


def test_transform_rejects_interval_layout_for_different_alpha() -> None:
    calibrator = SplitConformalCalibrator(alpha=0.2).fit(_predictions([1.0]), pd.Series([1.0]))

    with pytest.raises(ConfigurationError, match="quantile layout"):
        calibrator.transform(
            _predictions(
                [0.0, 1.0, 2.0],
                quantiles=[0.2, 0.5, 0.8],
                start="2026-02-01 00:00Z",
            )
        )


@pytest.mark.parametrize(
    "actuals",
    [
        pd.Series([1.0, 2.0], index=[1, 0]),
        pd.Series([1.0]),
        pd.Series([1.0, float("nan")]),
        pd.Series([1.0, True]),
        pd.Series([1.0, [2.0]]),
    ],
)
def test_fit_rejects_misaligned_or_nonfinite_actuals(actuals: pd.Series) -> None:
    with pytest.raises(ConfigurationError, match="actuals"):
        SplitConformalCalibrator(alpha=0.2).fit(_predictions([1.0, 2.0]), actuals)


def test_interval_actuals_align_to_groups_not_rows() -> None:
    calibrator = SplitConformalCalibrator(alpha=0.2).fit(
        _predictions([0.0, 1.0, 2.0], quantiles=[0.1, 0.5, 0.9]),
        pd.Series([1.0]),
    )

    assert calibrator.adjustment_ == -1.0
