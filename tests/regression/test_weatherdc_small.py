from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from climadc.reporting.artifacts import REQUIRED_ARTIFACTS
from examples.weatherdc_kasetsart.run import _last_available_value, run_small


def test_reference_persistence_uses_availability_not_only_event_time() -> None:
    origin = pd.Timestamp("2026-01-01 01:00:00+00:00")
    rows = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2026-01-01 00:00:00+00:00", "2026-01-01 01:00:00+00:00"]
            ),
            "available_at": pd.to_datetime(
                ["2026-01-01 00:00:00+00:00", "2026-01-01 02:00:00+00:00"]
            ),
            "value": [10.0, 999.0],
        }
    )

    assert _last_available_value(rows, origin) == 10.0


def test_weatherdc_small_is_causal_weather_aware_and_auditable(tmp_path: Path) -> None:
    result, run_dir = run_small(tmp_path / "runs")

    metrics = result.metrics["cooling_power"]
    assert metrics["weatherdc"]["mae"] < metrics["persistence"]["mae"]
    assert result.leakage_audit.rejected_rows == 0
    assert result.decision is not None
    assert result.decision.metrics["energy_conservation_error"] < 1e-8
    assert {path.name for path in run_dir.iterdir()} == REQUIRED_ARTIFACTS

    predictions = result.predictions.to_pandas()
    reference = predictions.loc[
        predictions["model_id"].isin({"weatherdc--split-000", "weatherdc-persistence--split-000"})
    ]
    assert set(reference["model_id"]) == {
        "weatherdc--split-000",
        "weatherdc-persistence--split-000",
    }
    actual = pd.read_csv(
        Path(__file__).parents[1] / "fixtures" / "weatherdc_small" / "expected.csv",
        parse_dates=["event_time"],
    ).set_index("event_time")["cooling_power"]
    for key, model_id in (
        ("weatherdc", "weatherdc--split-000"),
        ("persistence", "weatherdc-persistence--split-000"),
    ):
        rows = reference.loc[reference["model_id"] == model_id]
        errors = [
            abs(float(row.value) - float(actual.loc[row.valid_time]))
            for row in rows.itertuples(index=False)
        ]
        assert metrics[key]["mae"] == pytest.approx(sum(errors) / len(errors))
