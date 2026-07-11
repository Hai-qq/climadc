import warnings

import pandas as pd
import pytest

from climadc.contracts.frames import CLIMATE_COLUMNS, ClimateForecastFrame
from climadc.errors import ConfigurationError
from climadc.evaluation import (
    SliceAudit,
    SliceSizeWarning,
    audit_slice,
    extreme_weather_mask,
)


def _climate() -> ClimateForecastFrame:
    issue = pd.Timestamp("2026-01-01 00:00Z")
    rows = [
        ("dc-2", "humidity", 80.0, 4),
        ("dc-1", "temperature", 10.0, 1),
        ("dc-1", "humidity", 70.0, 2),
        ("dc-1", "temperature", 30.0, 3),
        ("dc-1", "temperature", 20.0, 2),
        ("dc-2", "temperature", 40.0, 4),
    ]
    frame = pd.DataFrame(
        {
            "site_id": [row[0] for row in rows],
            "issue_time": [issue] * len(rows),
            "available_at": [issue] * len(rows),
            "valid_time": [issue + pd.Timedelta(hours=row[3]) for row in rows],
            "variable": [row[1] for row in rows],
            "value": [row[2] for row in rows],
            "unit": ["percent" if row[1] == "humidity" else "degC" for row in rows],
            "source": ["fixture"] * len(rows),
            "quantile": [pd.NA] * len(rows),
            "member": [pd.NA] * len(rows),
        },
        columns=CLIMATE_COLUMNS,
    )
    return ClimateForecastFrame.from_pandas(frame)


def test_extreme_weather_mask_is_aligned_and_uses_higher_quantile() -> None:
    climate = _climate()
    normalized = climate.to_pandas()

    mask = extreme_weather_mask(climate, variable="temperature", quantile=0.5)

    assert mask.dtype == bool
    assert mask.index.equals(normalized.index)
    assert mask.tolist() == [False, False, False, True, False, True]
    assert normalized.loc[mask, "value"].tolist() == [30.0, 40.0]


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.1, True])
def test_extreme_weather_mask_rejects_invalid_quantile(quantile: object) -> None:
    with pytest.raises(ConfigurationError, match="quantile"):
        extreme_weather_mask(_climate(), "temperature", quantile)  # type: ignore[arg-type]


def test_extreme_weather_mask_rejects_missing_variable() -> None:
    with pytest.raises(ConfigurationError, match="No climate rows"):
        extreme_weather_mask(_climate(), "wind_speed", 0.9)


def test_audit_slice_returns_explicit_count_without_warning_when_usable() -> None:
    mask = pd.Series([True, False, True], dtype=bool)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        audit = audit_slice(mask, name="heat", minimum=2)

    assert audit == SliceAudit(name="heat", sample_count=2, minimum=2, usable=True)


def test_audit_slice_emits_one_structured_warning_when_too_small() -> None:
    mask = pd.Series([True, False, False], dtype=bool)

    with pytest.warns(SliceSizeWarning, match="heat.*1.*3") as caught:
        audit = audit_slice(mask, name="heat", minimum=3)

    assert len(caught) == 1
    warning = caught[0].message
    assert warning.name == "heat"
    assert warning.sample_count == 1
    assert warning.minimum == 3
    assert audit == SliceAudit(name="heat", sample_count=1, minimum=3, usable=False)


@pytest.mark.parametrize(
    ("mask", "minimum"),
    [
        (pd.Series([1, 0]), 1),
        (pd.Series([True, None], dtype="boolean"), 1),
        (pd.Series([True]), 0),
        (pd.Series([True]), True),
    ],
)
def test_audit_slice_validates_mask_and_minimum(mask: pd.Series, minimum: object) -> None:
    with pytest.raises(ConfigurationError):
        audit_slice(mask, name="heat", minimum=minimum)  # type: ignore[arg-type]


def test_audit_slice_requires_nonempty_name_and_series_mask() -> None:
    with pytest.raises(ConfigurationError, match="name"):
        audit_slice(pd.Series([True]), name="", minimum=1)
    with pytest.raises(ConfigurationError, match="pandas Series"):
        audit_slice([True], name="heat", minimum=1)  # type: ignore[arg-type]
