from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from climadc.adapters.sustaindc import SustainDCAdapter
from climadc.errors import ConfigurationError

START = pd.Timestamp("2026-08-01T00:00:00Z")


def _evaluation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day": [1, 1, 1],
            "hour": [0.0, 0.25, 0.5],
            "dc_ITE_total_power_kW": [500.0, 510.0, 520.0],
            "dc_HVAC_total_power_kW": [100.0, 101.0, 102.0],
            "dc_total_power_kW": [600.0, 611.0, 622.0],
            "outside_temp": [25.0, 25.5, 26.0],
            "bat_avg_CI": [300.0, 290.0, 280.0],
            "ls_shifted_workload": [0.5, 0.6, 0.7],
        }
    )


def test_sustaindc_converts_official_evaluation_columns_without_mutation() -> None:
    source = _evaluation_frame()
    original = source.copy(deep=True)
    result = SustainDCAdapter().from_pandas(
        source,
        site_id="sim-dc",
        region_id="sim-grid",
        start_time=START,
    )

    telemetry = result.telemetry.to_pandas()
    grid = result.grid_signals.to_pandas()
    assert_frame_equal(source, original)
    assert len(telemetry) == 12
    assert set(telemetry["metric"]) == {
        "it_power",
        "cooling_power",
        "total_power",
        "air_temperature",
    }
    assert set(telemetry.loc[telemetry["metric"] != "air_temperature", "unit"]) == {"kW"}
    assert set(telemetry.loc[telemetry["metric"] == "air_temperature", "unit"]) == {"degC"}
    assert (telemetry["available_at"] - telemetry["event_time"] == pd.Timedelta(minutes=15)).all()
    assert len(grid) == 3
    assert set(grid["value"]) == {280.0, 290.0, 300.0}
    assert set(grid["quality"]) == {"estimated"}
    assert grid["issue_time"].isna().all()
    assert result.metadata["availability_basis"] == "simulation interval end"
    assert "not converted into jobs" in result.metadata["workload_boundary"]


def test_sustaindc_reads_official_csv_shape(tmp_path: Path) -> None:
    path = tmp_path / "all_agents_episode_1.csv"
    _evaluation_frame().to_csv(path, index=False)
    result = SustainDCAdapter().read_evaluation(
        path,
        site_id="sim-dc",
        region_id="sim-grid",
        start_time=START,
        interval=pd.Timedelta(minutes=15),
    )
    assert len(result.telemetry.to_pandas()) == 12
    assert result.metadata["source"] == str(path.resolve())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns=["bat_avg_CI"]), "misses columns"),
        (
            lambda frame: frame.assign(hour=[0.0, 0.5, 0.25]),
            "unique and strictly ordered",
        ),
        (
            lambda frame: frame.assign(hour=[0.0, 0.25, 0.75]),
            "cadence",
        ),
        (
            lambda frame: frame.assign(dc_total_power_kW=[600.0, -1.0, 622.0]),
            "nonnegative",
        ),
        (
            lambda frame: frame.assign(outside_temp=[25.0, np.nan, 26.0]),
            "finite",
        ),
    ],
)
def test_sustaindc_rejects_incomplete_or_invalid_exports(mutation: object, message: str) -> None:
    assert callable(mutation)
    frame = mutation(_evaluation_frame())
    with pytest.raises(ConfigurationError, match=message):
        SustainDCAdapter().from_pandas(
            frame,
            site_id="sim-dc",
            region_id="sim-grid",
            start_time=START,
        )
