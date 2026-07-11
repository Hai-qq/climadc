from __future__ import annotations

import builtins

import pandas as pd
import pytest
import xarray as xr

from climadc.adapters.xarray import climate_from_xarray
from climadc.errors import ConfigurationError


def _dataset(*, timezone: str | None = "Asia/Shanghai") -> xr.Dataset:
    attrs = {} if timezone is None else {"timezone": timezone}
    return xr.Dataset(
        data_vars={
            "site": ("when", ["dc-1", "dc-1"]),
            "issued": ("when", [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")]),
            "retrieved": (
                "when",
                [pd.Timestamp("2026-01-01 00:05"), pd.Timestamp("2026-01-01 00:05")],
            ),
            "name": ("when", ["air_temperature", "air_temperature"]),
            "reading": ("when", [30.0, 31.0]),
            "units": ("when", ["degC", "degC"]),
        },
        coords={"when": [pd.Timestamp("2026-01-01 01:00"), pd.Timestamp("2026-01-01 02:00")]},
        attrs=attrs,
    )


MAPPING = {
    "site": "site_id",
    "issued": "issue_time",
    "retrieved": "available_at",
    "when": "valid_time",
    "name": "variable",
    "reading": "value",
    "units": "unit",
}


def test_climate_from_xarray_converts_dataset_without_stacking() -> None:
    result = climate_from_xarray(_dataset(), MAPPING, "forecast-archive").to_pandas()

    assert len(result) == 2
    assert set(result["source"]) == {"forecast-archive"}
    assert result["quantile"].isna().all()
    assert result["member"].isna().all()
    assert result.loc[0, "valid_time"] == pd.Timestamp("2025-12-31 17:00", tz="UTC")


def test_climate_from_xarray_accepts_canonical_columns_with_empty_mapping() -> None:
    canonical = _dataset().rename(MAPPING)

    result = climate_from_xarray(canonical, {}, "forecast-archive").to_pandas()

    assert len(result) == 2
    assert set(result["source"]) == {"forecast-archive"}


def test_climate_from_xarray_rejects_noncanonical_columns_with_empty_mapping() -> None:
    with pytest.raises(ConfigurationError, match=r"column_map.*xarray dataset"):
        climate_from_xarray(_dataset(), {}, "forecast-archive")


def test_climate_from_xarray_requires_timezone_for_naive_times() -> None:
    with pytest.raises(ConfigurationError, match="timezone"):
        climate_from_xarray(_dataset(timezone=None), MAPPING, "fixture")


def test_climate_from_xarray_requires_dataset() -> None:
    with pytest.raises(ConfigurationError, match="xarray.Dataset"):
        climate_from_xarray(object(), MAPPING, "fixture")


def test_climate_from_xarray_reports_only_missing_top_level_xarray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing_xarray(name: str, *args: object, **kwargs: object) -> object:
        if name == "xarray":
            raise ModuleNotFoundError("No module named 'xarray'", name="xarray")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_xarray)

    with pytest.raises(ConfigurationError) as exc_info:
        climate_from_xarray(object(), {}, "fixture")
    assert str(exc_info.value) == "Install climadc[xarray]"


def test_climate_from_xarray_propagates_unrelated_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def broken_dependency(name: str, *args: object, **kwargs: object) -> object:
        if name == "xarray":
            raise ModuleNotFoundError("No module named 'dependency'", name="dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_dependency)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        climate_from_xarray(object(), {}, "fixture")
    assert exc_info.value.name == "dependency"
