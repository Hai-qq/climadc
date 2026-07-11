from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from climadc.errors import ConfigurationError
import climadc.forecasting.lightgbm as lightgbm_module
from climadc.forecasting.lightgbm import LightGBMForecaster


def _training_frame() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "site_id": ["dc-1"] * len(times),
            "valid_time": times,
            "available_at": times,
            "target": ["power"] * len(times),
            "value": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "unit": ["kW"] * len(times),
        }
    )


def test_lightgbm_constructor_reports_exact_extra_install_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def import_without_lightgbm(name: str, package: str | None = None) -> object:
        if name == "lightgbm":
            raise ModuleNotFoundError(name="lightgbm")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_without_lightgbm)

    with pytest.raises(ConfigurationError) as exc_info:
        LightGBMForecaster(target="power", features=("hour",))

    assert str(exc_info.value) == "Install climadc[lightgbm]"


@pytest.mark.parametrize(
    "error",
    [
        OSError("Library not loaded: @rpath/libomp.dylib"),
        OSError("libgomp.so.1: cannot open shared object file: No such file or directory"),
        ImportError("DLL load failed while importing basic: vcomp140.dll was not found"),
        ImportError("DLL load failed while importing basic: specified module was not found"),
        OSError("cannot open shared object file: No such file or directory"),
    ],
)
def test_lightgbm_constructor_classifies_recognized_native_loader_absence(
    monkeypatch: pytest.MonkeyPatch, error: ImportError | OSError
) -> None:
    def fail_import(name: str, package: str | None = None) -> object:
        raise error

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(ConfigurationError) as exc_info:
        LightGBMForecaster(target="power", features=("hour",))

    assert str(exc_info.value) == "Install climadc[lightgbm]"
    assert exc_info.value.__cause__ is error
    assert lightgbm_module._is_lightgbm_unavailable(error)


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("nested dependency missing", name="numpy"),
        ImportError("unexpected LightGBM internal API mismatch"),
        OSError("permission denied while reading an unrelated file"),
    ],
)
def test_lightgbm_constructor_propagates_unrelated_import_failures(
    monkeypatch: pytest.MonkeyPatch, error: ImportError | OSError
) -> None:
    def fail_import(name: str, package: str | None = None) -> object:
        raise error

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(type(error)) as exc_info:
        LightGBMForecaster(target="power", features=("hour",))

    assert exc_info.value is error
    assert not lightgbm_module._is_lightgbm_unavailable(error)


class _FakeRegressor:
    def __init__(self, **kwargs: object) -> None:
        self._training_mean = 0.0

    def fit(self, features: pd.DataFrame, target: Any) -> "_FakeRegressor":
        self._training_mean = float(features["elapsed_hours"].mean())
        return self

    def predict(self, features: pd.DataFrame) -> list[float]:
        return [float(value) + self._training_mean for value in features["elapsed_hours"]]


@pytest.fixture
def fake_lightgbm(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    def import_fake_lightgbm(name: str, package: str | None = None) -> object:
        if name == "lightgbm":
            return SimpleNamespace(LGBMRegressor=_FakeRegressor)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_fake_lightgbm)


def _cross_site_history(include_earlier_site_b: bool) -> pd.DataFrame:
    rows = [
        (site, timestamp, timestamp, "power", base + offset, "kW")
        for site, base in (("dc-1", 100.0), ("dc-2", 200.0))
        for timestamp, offset in (
            ("2026-01-02 00:00Z", 0.0),
            ("2026-01-02 06:00Z", 6.0),
            ("2026-01-03 00:00Z", 24.0),
            ("2026-01-03 06:00Z", 30.0),
        )
    ]
    if include_earlier_site_b:
        rows.append(
            (
                "dc-2",
                "2026-01-01 00:00Z",
                "2026-01-01 00:00Z",
                "power",
                190.0,
                "kW",
            )
        )
    frame = pd.DataFrame(
        rows,
        columns=["site_id", "valid_time", "available_at", "target", "value", "unit"],
    )
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    return frame


def test_lightgbm_elapsed_anchor_and_site_prediction_are_cross_site_isolated(
    fake_lightgbm: None,
) -> None:
    baseline = LightGBMForecaster(target="power", features=("elapsed_hours",)).fit(
        _cross_site_history(include_earlier_site_b=False), context={}
    )
    perturbed = LightGBMForecaster(target="power", features=("elapsed_hours",)).fit(
        _cross_site_history(include_earlier_site_b=True), context={}
    )
    origins = pd.DatetimeIndex(["2026-01-04 00:00Z"])

    assert baseline.feature_anchors_["dc-1"] == pd.Timestamp("2026-01-02 00:00Z")
    assert perturbed.feature_anchors_["dc-1"] == baseline.feature_anchors_["dc-1"]
    baseline_value = (
        baseline.predict(origins, pd.Timedelta("6h"))
        .to_pandas()
        .query("site_id == 'dc-1'")["value"]
    )
    perturbed_value = (
        perturbed.predict(origins, pd.Timedelta("6h"))
        .to_pandas()
        .query("site_id == 'dc-1'")["value"]
    )
    assert perturbed_value.tolist() == pytest.approx(baseline_value.tolist())


def test_lightgbm_failed_refit_preserves_previous_fitted_state(fake_lightgbm: None) -> None:
    model = LightGBMForecaster(target="power", features=("elapsed_hours",)).fit(
        _cross_site_history(include_earlier_site_b=False), context={}
    )
    origins = pd.DatetimeIndex(["2026-01-04 00:00Z"])
    before = model.predict(origins, pd.Timedelta("6h")).to_pandas()
    old_models = model.models_.copy()
    old_anchors = model.feature_anchors_.copy()
    invalid = _cross_site_history(include_earlier_site_b=False)
    invalid = invalid.loc[~((invalid["site_id"] == "dc-2") & (invalid.index != 4))]

    with pytest.raises(ConfigurationError, match="Insufficient training rows"):
        model.fit(invalid, context={})

    assert model.models_ == old_models
    assert model.feature_anchors_ == old_anchors
    assert_frame_equal(model.predict(origins, pd.Timedelta("6h")).to_pandas(), before)


@pytest.mark.lightgbm
def test_lightgbm_installed_path_uses_calendar_features_and_prediction_contract() -> None:
    try:
        model = LightGBMForecaster(target="power", features=("hour", "dayofweek"))
    except ConfigurationError as exc:
        if lightgbm_module._is_lightgbm_unavailable(exc.__cause__):
            pytest.skip(f"LightGBM native library is unavailable: {exc.__cause__}")
        raise

    model.fit(_training_frame(), context={})
    frame = model.predict(pd.DatetimeIndex(["2026-01-02 00:00Z"]), pd.Timedelta("1h")).to_pandas()

    assert model.features == ("hour", "dayofweek")
    assert frame.loc[0, "site_id"] == "dc-1"
    assert frame.loc[0, "model_id"] == "lightgbm"
    assert frame.loc[0, "valid_time"] == pd.Timestamp("2026-01-02 01:00Z")
