from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from climadc.contracts.frames import PREDICTION_COLUMNS, PredictionFrame
from climadc.errors import ConfigurationError
from climadc.forecasting import (
    ClimatologyForecaster,
    LinearForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)


def _history(rows: list[tuple[str, str, str, str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["site_id", "valid_time", "available_at", "target", "value", "unit"],
    ).assign(
        valid_time=lambda frame: pd.to_datetime(frame["valid_time"], utc=True),
        available_at=lambda frame: pd.to_datetime(frame["available_at"], utc=True),
        declared_feature=1.0,
    )


@pytest.fixture
def persistence_history() -> pd.DataFrame:
    return _history(
        [
            ("dc-1", "2026-01-01 00:00Z", "2026-01-01 00:00Z", "power", 100.0, "kW"),
            ("dc-1", "2026-01-01 01:00Z", "2026-01-01 01:00Z", "power", 101.0, "kW"),
            ("dc-1", "2026-01-01 01:00Z", "2026-01-01 01:00Z", "power", 102.0, "kW"),
            ("dc-1", "2026-01-01 02:00Z", "2026-01-01 05:00Z", "power", 999.0, "kW"),
            ("dc-2", "2026-01-01 00:00Z", "2026-01-01 00:30Z", "power", 200.0, "kW"),
            ("dc-2", "2026-01-01 01:30Z", "2026-01-01 01:30Z", "power", 201.0, "kW"),
            ("dc-1", "2026-01-01 00:00Z", "2026-01-01 00:00Z", "temperature", 20.0, "degC"),
        ]
    )


def test_persistence_returns_latest_legal_value_without_future_leakage(
    persistence_history: pd.DataFrame,
) -> None:
    model = PersistenceForecaster(target="power", model_id="persist-v1")
    fitted = model.fit(persistence_history, context={"ignored": object()})

    result = fitted.predict(
        pd.DatetimeIndex(["2026-01-01 01:00Z", "2026-01-01 02:00Z"]),
        pd.Timedelta("4h"),
    )
    frame = result.to_pandas()

    assert fitted is model
    assert isinstance(result, PredictionFrame)
    assert list(frame.columns) == list(PREDICTION_COLUMNS)
    assert frame[["site_id", "issue_time"]].values.tolist() == [
        ["dc-1", pd.Timestamp("2026-01-01 01:00Z")],
        ["dc-1", pd.Timestamp("2026-01-01 02:00Z")],
        ["dc-2", pd.Timestamp("2026-01-01 01:00Z")],
        ["dc-2", pd.Timestamp("2026-01-01 02:00Z")],
    ]
    assert frame["value"].tolist() == [102.0, 102.0, 200.0, 201.0]
    assert (frame["valid_time"] - frame["issue_time"] == pd.Timedelta("4h")).all()
    assert frame["quantile"].isna().all()
    assert set(frame["model_id"]) == {"persist-v1"}
    assert model.target == "power"
    assert model.unit == "kW"


def test_persistence_requires_legal_history_for_every_site_origin(
    persistence_history: pd.DataFrame,
) -> None:
    model = PersistenceForecaster(target="power").fit(persistence_history, context={})

    with pytest.raises(ConfigurationError, match="No legal history"):
        model.predict(pd.DatetimeIndex(["2025-12-31 23:00Z"]), pd.Timedelta("1h"))


def test_fit_does_not_mutate_training_data_and_normalizes_aware_timestamps(
    persistence_history: pd.DataFrame,
) -> None:
    local = persistence_history.copy(deep=True)
    local["valid_time"] = local["valid_time"].dt.tz_convert("Asia/Shanghai")
    local["available_at"] = local["available_at"].dt.tz_convert("Asia/Shanghai")
    before = local.copy(deep=True)

    model = PersistenceForecaster(target="power").fit(local, context={})

    assert_frame_equal(local, before)
    assert str(model._train["valid_time"].dtype).endswith("UTC]")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda frame: frame.assign(valid_time=frame["valid_time"].dt.tz_localize(None)),
            "timezone-aware",
        ),
        (lambda frame: frame.assign(value=float("inf")), "finite real"),
        (lambda frame: frame.assign(site_id=""), "site_id"),
        (lambda frame: frame.assign(target=""), "target"),
        (lambda frame: frame.assign(unit=""), "unit"),
    ],
)
def test_fit_rejects_invalid_training_rows(
    persistence_history: pd.DataFrame,
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
    match: str,
) -> None:
    with pytest.raises(ConfigurationError, match=match):
        PersistenceForecaster(target="power").fit(mutate(persistence_history), context={})


def test_fit_requires_selected_target_and_exactly_one_unit(
    persistence_history: pd.DataFrame,
) -> None:
    with pytest.raises(ConfigurationError, match="No training rows for target"):
        PersistenceForecaster(target="missing").fit(persistence_history, context={})

    mixed = persistence_history.copy(deep=True)
    mixed.loc[1, "unit"] = "W"
    with pytest.raises(ConfigurationError, match="exactly one unit"):
        PersistenceForecaster(target="power").fit(mixed, context={})


@pytest.mark.parametrize(
    "model",
    [
        PersistenceForecaster(target="power"),
        SeasonalNaiveForecaster(target="power", period=pd.Timedelta("24h")),
        ClimatologyForecaster(target="power", group_by=()),
        LinearForecaster(target="power", features=("hour",)),
    ],
)
def test_predict_before_fit_fails(model: Any) -> None:
    with pytest.raises(ConfigurationError, match="not fitted"):
        model.predict(pd.DatetimeIndex(["2026-01-01 00:00Z"]), pd.Timedelta("1h"))


@pytest.mark.parametrize(
    ("origins", "horizon", "match"),
    [
        (pd.DatetimeIndex([]), pd.Timedelta("1h"), "non-empty"),
        (pd.DatetimeIndex(["2026-01-01"]), pd.Timedelta("1h"), "timezone-aware"),
        (
            pd.DatetimeIndex([pd.Timestamp("2026-01-01", tz="Asia/Shanghai")]),
            pd.Timedelta("1h"),
            "exact UTC",
        ),
        (
            pd.DatetimeIndex(["2026-01-01 01:00Z", "2026-01-01 00:00Z"]),
            pd.Timedelta("1h"),
            "sorted",
        ),
        (
            pd.DatetimeIndex(["2026-01-01 00:00Z", "2026-01-01 00:00Z"]),
            pd.Timedelta("1h"),
            "unique",
        ),
        (pd.DatetimeIndex(["2026-01-01 00:00Z"]), pd.Timedelta(0), "positive"),
    ],
)
def test_predict_rejects_invalid_origins_or_horizon(
    persistence_history: pd.DataFrame,
    origins: pd.DatetimeIndex,
    horizon: pd.Timedelta,
    match: str,
) -> None:
    model = PersistenceForecaster(target="power").fit(persistence_history, context={})

    with pytest.raises(ConfigurationError, match=match):
        model.predict(origins, horizon)


def test_seasonal_naive_uses_exact_available_reference_without_persistence_fallback() -> None:
    train = _history(
        [
            ("dc-1", "2026-01-01 03:00Z", "2026-01-01 03:00Z", "power", 90.0, "kW"),
            ("dc-1", "2026-01-01 04:00Z", "2026-01-01 05:00Z", "power", 100.0, "kW"),
            ("dc-1", "2026-01-01 04:00Z", "2026-01-02 01:00Z", "power", 999.0, "kW"),
        ]
    )
    model = SeasonalNaiveForecaster(
        target="power", period=pd.Timedelta("24h"), model_id="seasonal-v1"
    ).fit(train, context={})

    result = model.predict(pd.DatetimeIndex(["2026-01-02 00:00Z"]), pd.Timedelta("4h")).to_pandas()
    assert result["value"].tolist() == [100.0]
    assert model.period == pd.Timedelta("24h")

    with pytest.raises(ConfigurationError, match="No exact seasonal reference"):
        model.predict(pd.DatetimeIndex(["2026-01-02 01:00Z"]), pd.Timedelta("4h"))


@pytest.mark.parametrize("period", [pd.Timedelta(0), pd.Timedelta("-1h"), "24h"])
def test_seasonal_naive_requires_explicit_positive_timedelta(period: object) -> None:
    with pytest.raises(ConfigurationError, match="period.*positive"):
        SeasonalNaiveForecaster(target="power", period=period)  # type: ignore[arg-type]


def test_climatology_uses_only_declared_derived_groups() -> None:
    train = _history(
        [
            ("dc-1", "2026-01-01 04:00Z", "2026-01-01 04:00Z", "power", 100.0, "kW"),
            ("dc-1", "2026-01-02 04:00Z", "2026-01-02 04:00Z", "power", 120.0, "kW"),
            ("dc-2", "2026-01-01 04:00Z", "2026-01-01 04:00Z", "power", 200.0, "kW"),
            ("dc-2", "2026-01-02 04:00Z", "2026-01-02 04:00Z", "power", 240.0, "kW"),
        ]
    )
    model = ClimatologyForecaster(
        target="power", group_by=("site_id", "hour"), model_id="clim-v1"
    ).fit(train, context={})

    frame = model.predict(pd.DatetimeIndex(["2026-01-03 00:00Z"]), pd.Timedelta("4h")).to_pandas()
    assert frame["value"].tolist() == [110.0, 220.0]
    assert model.group_by == ("site_id", "hour")

    with pytest.raises(ConfigurationError, match="No climatology group"):
        model.predict(pd.DatetimeIndex(["2026-01-03 01:00Z"]), pd.Timedelta("4h"))


def test_climatology_supports_explicit_global_group() -> None:
    train = _history(
        [
            ("dc-1", "2026-01-01 00:00Z", "2026-01-01 00:00Z", "power", 100.0, "kW"),
            ("dc-2", "2026-01-01 01:00Z", "2026-01-01 01:00Z", "power", 200.0, "kW"),
        ]
    )
    model = ClimatologyForecaster(target="power", group_by=()).fit(train, context={})

    frame = model.predict(pd.DatetimeIndex(["2026-01-02 00:00Z"]), pd.Timedelta("1h")).to_pandas()
    assert frame["value"].tolist() == [150.0, 150.0]


@pytest.mark.parametrize(
    "group_by",
    [("weekday",), ("hour", "hour")],
)
def test_climatology_rejects_unknown_or_duplicate_groups(group_by: tuple[str, ...]) -> None:
    with pytest.raises(ConfigurationError, match="group_by"):
        ClimatologyForecaster(target="power", group_by=group_by)


def test_linear_preserves_feature_order_and_fits_required_pipeline_per_site() -> None:
    rows = [
        (
            site,
            f"2026-01-0{day} {hour:02d}:00Z",
            f"2026-01-0{day} {hour:02d}:00Z",
            "power",
            base + hour,
            "kW",
        )
        for site, base in (("dc-1", 100.0), ("dc-2", 200.0))
        for day, hour in ((1, 0), (1, 6), (1, 12), (2, 0), (2, 6), (2, 12))
    ]
    train = _history(rows)
    features = ("dayofweek", "hour", "elapsed_hours")
    model = LinearForecaster(target="power", features=features, model_id="ridge-v1")

    model.fit(train, context={})
    frame = model.predict(pd.DatetimeIndex(["2026-01-03 00:00Z"]), pd.Timedelta("6h")).to_pandas()

    assert model.features == features
    assert set(model.models_) == {"dc-1", "dc-2"}
    for pipeline in model.models_.values():
        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps) == ["scale", "model"]
        assert isinstance(pipeline.named_steps["scale"], StandardScaler)
        assert isinstance(pipeline.named_steps["model"], Ridge)
    assert len(frame) == 2
    assert frame["value"].map(lambda value: isinstance(value, float)).all()


@pytest.mark.parametrize(
    "features",
    [(), ("weekday",), ("hour", "hour")],
)
def test_linear_rejects_missing_unknown_or_duplicate_features(
    features: tuple[str, ...],
) -> None:
    with pytest.raises(ConfigurationError, match="features"):
        LinearForecaster(target="power", features=features)


def test_linear_rejects_site_with_insufficient_rows() -> None:
    train = _history([("dc-1", "2026-01-01 00:00Z", "2026-01-01 00:00Z", "power", 100.0, "kW")])

    with pytest.raises(ConfigurationError, match="Insufficient training rows"):
        LinearForecaster(target="power", features=("hour",)).fit(train, context={})
