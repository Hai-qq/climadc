from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from climadc.adapters.local import read_climate, read_telemetry, read_workload
from climadc.errors import ConfigurationError, ContractError


def _climate_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site": ["dc-1"],
            "issued": [pd.Timestamp("2026-01-01 00:00")],
            "retrieved": [pd.Timestamp("2026-01-01 00:05")],
            "valid": [pd.Timestamp("2026-01-01 04:00")],
            "name": ["air_temperature"],
            "reading": [30.0],
            "units": ["degC"],
            "provider": ["fixture"],
            "quantile": [pd.NA],
            "member": [pd.NA],
        }
    )


CLIMATE_MAP = {
    "site": "site_id",
    "issued": "issue_time",
    "retrieved": "available_at",
    "valid": "valid_time",
    "name": "variable",
    "reading": "value",
    "units": "unit",
    "provider": "source",
}


def _telemetry_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site": ["dc-1"],
            "device": ["meter-1"],
            "event": [pd.Timestamp("2026-01-01 00:00", tz="Asia/Shanghai")],
            "available": [pd.Timestamp("2026-01-01 00:05", tz="Asia/Shanghai")],
            "name": ["total_power"],
            "reading": [100.0],
            "units": ["kW"],
            "status": ["observed"],
        }
    )


TELEMETRY_MAP = {
    "site": "site_id",
    "device": "device_id",
    "event": "event_time",
    "available": "available_at",
    "name": "metric",
    "reading": "value",
    "units": "unit",
    "status": "quality",
}


def test_read_climate_normalizes_equivalent_csv_and_parquet(tmp_path: Path) -> None:
    source = _climate_source()
    csv_path = tmp_path / "climate.csv"
    parquet_path = tmp_path / "climate.parquet"
    source.to_csv(csv_path, index=False)
    source.to_parquet(parquet_path, index=False)

    csv = read_climate(csv_path, "csv", CLIMATE_MAP, "Asia/Shanghai").to_pandas()
    parquet = read_climate(parquet_path, "parquet", CLIMATE_MAP, "Asia/Shanghai").to_pandas()

    pd.testing.assert_frame_equal(csv, parquet)
    assert csv.loc[0, "issue_time"] == pd.Timestamp("2025-12-31 16:00", tz="UTC")


def test_read_telemetry_renames_source_columns_and_converts_aware_times(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.parquet"
    _telemetry_source().to_parquet(path, index=False)

    frame = read_telemetry(path, "parquet", TELEMETRY_MAP, "UTC").to_pandas()

    assert frame.loc[0, "event_time"] == pd.Timestamp("2025-12-31 16:00", tz="UTC")
    assert frame.loc[0, "metric"] == "total_power"


def test_read_telemetry_normalizes_equivalent_csv_and_parquet_with_null_device(
    tmp_path: Path,
) -> None:
    source = _telemetry_source()
    source["device"] = pd.NA
    csv_path = tmp_path / "telemetry.csv"
    parquet_path = tmp_path / "telemetry.parquet"
    source.to_csv(csv_path, index=False)
    source.to_parquet(parquet_path, index=False)

    csv = read_telemetry(csv_path, "csv", TELEMETRY_MAP, "UTC").to_pandas()
    parquet = read_telemetry(parquet_path, "parquet", TELEMETRY_MAP, "UTC").to_pandas()

    pd.testing.assert_frame_equal(csv, parquet)
    assert csv["device_id"].isna().all()
    assert csv["device_id"].dtype == object


def test_empty_column_map_accepts_only_already_canonical_input(tmp_path: Path) -> None:
    canonical = _climate_source().rename(columns=CLIMATE_MAP)
    accepted = tmp_path / "canonical.csv"
    rejected = tmp_path / "source.csv"
    canonical.to_csv(accepted, index=False)
    _climate_source().to_csv(rejected, index=False)

    result = read_climate(accepted, "csv", {}, "UTC")
    assert result.to_pandas().columns.tolist() == canonical.columns.tolist()

    with pytest.raises(ConfigurationError, match=str(rejected)):
        read_climate(rejected, "csv", {}, "UTC")


@pytest.mark.parametrize(
    "column_map",
    [
        {"missing": "site_id"},
        {"site": "site_id", "issued": "site_id"},
        {"site": "issued"},
    ],
)
def test_read_local_rejects_missing_duplicate_and_colliding_mappings(
    tmp_path: Path, column_map: dict[str, str]
) -> None:
    path = tmp_path / "climate.csv"
    _climate_source().to_csv(path, index=False)

    with pytest.raises(ConfigurationError, match=str(path)):
        read_climate(path, "csv", column_map, "UTC")


@pytest.mark.parametrize(
    ("timestamp", "timezone"),
    [
        ("2025-11-02 01:30", "America/New_York"),
        ("2025-03-09 02:30", "America/New_York"),
        ("2026-01-01 00:00", "Not/A_Timezone"),
    ],
)
def test_read_local_rejects_ambiguous_nonexistent_or_invalid_timezone(
    tmp_path: Path, timestamp: str, timezone: str
) -> None:
    source = _climate_source()
    source["issued"] = [timestamp]
    source["retrieved"] = [timestamp]
    source["valid"] = [timestamp]
    path = tmp_path / "dst.csv"
    source.to_csv(path, index=False)

    with pytest.raises(ConfigurationError, match=str(path)):
        read_climate(path, "csv", CLIMATE_MAP, timezone)


@pytest.mark.parametrize("format", ["json", "CSV"])
def test_read_local_rejects_unsupported_formats_with_path(tmp_path: Path, format: str) -> None:
    path = tmp_path / "climate.data"
    path.write_text("not used", encoding="utf-8")

    with pytest.raises(ConfigurationError, match=str(path)):
        read_climate(path, format, CLIMATE_MAP, "UTC")  # type: ignore[arg-type]


def test_read_local_wraps_file_errors_with_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"

    with pytest.raises(ConfigurationError, match=str(path)):
        read_climate(path, "csv", CLIMATE_MAP, "UTC")


def test_read_local_delegates_canonical_value_validation(tmp_path: Path) -> None:
    source = _climate_source()
    source.loc[0, "reading"] = float("inf")
    path = tmp_path / "invalid.csv"
    source.to_csv(path, index=False)

    with pytest.raises(ContractError, match="value must be finite"):
        read_climate(path, "csv", CLIMATE_MAP, "UTC")


def test_read_local_wraps_non_scalar_timestamp_with_path(tmp_path: Path) -> None:
    source = _climate_source()
    source["issued"] = pd.Series([["2026-01-01", "2026-01-02"]], dtype=object)
    path = tmp_path / "invalid-time.parquet"
    source.to_parquet(path, index=False)

    with pytest.raises(ConfigurationError, match=str(path)):
        read_climate(path, "parquet", CLIMATE_MAP, "UTC")


def test_read_workload_reuses_local_normalization(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "job": ["batch-1"],
            "site": ["dc-1"],
            "event": ["2026-01-01 08:00"],
            "available": ["2026-01-01 08:00"],
            "deadline_at": ["2026-01-01 10:00"],
            "resource": ["compute"],
            "amount": [4.0],
            "units": ["kWh"],
            "flex": [0.5],
        }
    )
    path = tmp_path / "workload.csv"
    source.to_csv(path, index=False)

    result = read_workload(
        path,
        "csv",
        {
            "job": "job_id",
            "site": "site_id",
            "event": "event_time",
            "available": "available_at",
            "deadline_at": "deadline",
            "resource": "resource_type",
            "amount": "demand",
            "units": "unit",
            "flex": "flexible_fraction",
        },
        "Asia/Shanghai",
    ).to_pandas()

    assert result.loc[0, "event_time"] == pd.Timestamp("2026-01-01 00:00Z")
    assert result.loc[0, "deadline"] == pd.Timestamp("2026-01-01 02:00Z")
