from __future__ import annotations

import importlib

import pandas as pd
import pytest

from climadc.errors import ConfigurationError
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


@pytest.mark.lightgbm
def test_lightgbm_installed_path_uses_calendar_features_and_prediction_contract() -> None:
    try:
        pytest.importorskip("lightgbm")
    except OSError as exc:
        pytest.skip(f"LightGBM native library is unavailable: {exc}")
    model = LightGBMForecaster(target="power", features=("hour", "dayofweek"))

    model.fit(_training_frame(), context={})
    frame = model.predict(pd.DatetimeIndex(["2026-01-02 00:00Z"]), pd.Timedelta("1h")).to_pandas()

    assert model.features == ("hour", "dayofweek")
    assert frame.loc[0, "site_id"] == "dc-1"
    assert frame.loc[0, "model_id"] == "lightgbm"
    assert frame.loc[0, "valid_time"] == pd.Timestamp("2026-01-02 01:00Z")
