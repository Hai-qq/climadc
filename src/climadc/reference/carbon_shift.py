from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import cast

import pandas as pd
import yaml

from climadc.adapters.neso import NESOCarbonIntensityAdapter
from climadc.adapters.openmeteo_history import OpenMeteoHistoryAdapter
from climadc.contracts import FlexibleWorkloadFrame, GridSignalFrame
from climadc.errors import ConfigurationError
from climadc.replay.manifest import SourceManifest, SourceRecord, SourceTiming, sha256_file

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gb_london_24h"
_SITE_ID = "gb-london-reference"
_LATITUDE = 51.5074
_LONGITUDE = -0.1278
_PROJECT_URL = (
    "https://github.com/Hai-qq/climadc/tree/main/src/climadc/reference/fixtures/gb_london_24h"
)


def packaged_study_path() -> Path:
    path = _FIXTURE_DIR / "study.yaml"
    if not path.is_file():
        raise ConfigurationError("Packaged carbon-shift reference fixture is missing")
    return path.resolve()


def packaged_suite_path() -> Path:
    path = _FIXTURE_DIR / "suite.yaml"
    if not path.is_file():
        raise ConfigurationError("Packaged replay robustness suite is missing")
    return path.resolve()


def _tariff_value(hour: int) -> float:
    if hour < 6:
        return 0.12
    if hour < 16:
        return 0.25
    if hour < 20:
        return 0.45
    return 0.18


def _tariff(decision_time: pd.Timestamp, horizon: pd.Timedelta) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    slots = pd.date_range(decision_time, periods=int(horizon / pd.Timedelta(hours=1)), freq="1h")
    for slot in slots:
        value = _tariff_value(slot.hour)
        rows.extend(
            [
                {
                    "site_id": _SITE_ID,
                    "region_id": "GB",
                    "issue_time": decision_time,
                    "available_at": decision_time,
                    "valid_time": slot,
                    "signal": "energy_price",
                    "value": value,
                    "unit": "GBP / kWh",
                    "source": "declared-gb-time-of-use-scenario",
                    "quality": "forecast",
                    "quantile": pd.NA,
                },
                {
                    "site_id": _SITE_ID,
                    "region_id": "GB",
                    "issue_time": pd.NaT,
                    "available_at": slot + pd.Timedelta(hours=1),
                    "valid_time": slot,
                    "signal": "energy_price",
                    "value": value,
                    "unit": "GBP / kWh",
                    "source": "declared-gb-time-of-use-scenario",
                    "quality": "estimated",
                    "quantile": pd.NA,
                },
            ]
        )
    return cast(pd.DataFrame, pd.DataFrame(rows))


def _workload(decision_time: pd.Timestamp) -> FlexibleWorkloadFrame:
    rows = pd.DataFrame(
        {
            "job_id": ["batch-urgent", "batch-morning", "batch-afternoon", "batch-daily"],
            "site_id": [_SITE_ID] * 4,
            "release_time": [decision_time] * 4,
            "available_at": [decision_time] * 4,
            "deadline": [
                decision_time + pd.Timedelta(hours=8),
                decision_time + pd.Timedelta(hours=12),
                decision_time + pd.Timedelta(hours=18),
                decision_time + pd.Timedelta(hours=24),
            ],
            "energy": [400.0, 300.0, 400.0, 500.0],
            "energy_unit": ["kWh"] * 4,
            "max_power": [100.0, 75.0, 100.0, 125.0],
            "power_unit": ["kW"] * 4,
            "preemptible": [True] * 4,
            "priority": [3.0, 2.0, 1.0, 0.0],
        }
    )
    return FlexibleWorkloadFrame.from_pandas(cast(pd.DataFrame, rows))


def _study_payload(decision_time: pd.Timestamp) -> dict[str, object]:
    return {
        "schema_version": "1",
        "study_id": "gb-london-carbon-shift-24h",
        "decision_time": decision_time.isoformat(),
        "inputs": {
            "climate_forecast": {
                "path": "climate-forecast.csv",
                "format": "csv",
                "timezone": "UTC",
            },
            "actual_weather": {
                "path": "actual-weather.csv",
                "format": "csv",
                "timezone": "UTC",
            },
            "grid_signals": {
                "path": "grid-signals.csv",
                "format": "csv",
                "timezone": "UTC",
            },
            "workload": {"path": "workload.csv", "format": "csv", "timezone": "UTC"},
        },
        "source_manifest": "source-manifest.yaml",
        "replay": {
            "site_id": _SITE_ID,
            "horizon": "24h",
            "interval": "1h",
            "it_capacity_kw": 500,
            "fixed_it_power_kw": 300,
            "cost_weight": 1,
            "carbon_weight": 1,
            "demand_charge_per_kw": 0,
            "tolerance_kwh": 1e-7,
        },
        "facility_model": {
            "kind": "temperature_sensitive_pue",
            "reference_temperature_c": 18,
            "base_pue": 1.2,
            "slope_per_degree_c": 0.015,
            "min_pue": 1.1,
            "max_pue": 1.8,
        },
        "assumptions": {
            "location": "London, United Kingdom",
            "latitude": _LATITUDE,
            "longitude": _LONGITUDE,
            "tariff": "Declared time-of-use scenario; not a supplier tariff or observed bill",
            "weather_forecast": "Open-Meteo previous_day1 fixed 24-hour lead",
            "weather_settlement": "Open-Meteo gridded historical model estimate",
            "carbon_scope": "National Great Britain signal applied to a London site",
            "workload": "Deterministic project-owned batch-job fixture",
        },
        "limitations": [
            "Weather settlement is a gridded model estimate, not measured site telemetry.",
            (
                "Historical API payloads do not prove forecast availability at the replay "
                "decision time; availability is declared as a scenario assumption."
            ),
            "The national GB carbon signal does not resolve London regional variation.",
            (
                "The tariff and workload are synthetic scenario inputs, so results are not "
                "production savings."
            ),
        ],
        "output_dir": "replay-runs",
    }


def _record(
    *,
    source_id: str,
    artifact: Path,
    selection: str,
    role: str,
    provider: str,
    request_url: str,
    retrieved_at: pd.Timestamp,
    license_name: str,
    attribution: str,
    provenance: str,
    transformations: list[str],
    timing: SourceTiming,
    limitations: list[str],
) -> SourceRecord:
    return SourceRecord.model_validate(
        {
            "source_id": source_id,
            "artifact": artifact.name,
            "selection": selection,
            "role": role,
            "provider": provider,
            "request_url": request_url,
            "retrieved_at": retrieved_at.isoformat(),
            "sha256": sha256_file(artifact),
            "bytes": artifact.stat().st_size,
            "license": license_name,
            "attribution": attribution,
            "provenance": provenance,
            "transformations": transformations,
            "timing": timing.model_dump(mode="json"),
            "limitations": limitations,
        }
    )


def _manifest(
    directory: Path,
    *,
    retrieved_at: pd.Timestamp,
    weather_metadata: dict[str, str],
    carbon_metadata: dict[str, str],
) -> SourceManifest:
    climate = directory / "climate-forecast.csv"
    actual = directory / "actual-weather.csv"
    grid = directory / "grid-signals.csv"
    workload = directory / "workload.csv"
    records = [
        _record(
            source_id="open-meteo-weather-forecast",
            artifact=climate,
            selection="variable=air_temperature; quality=24h fixed-lead forecast",
            role="issued weather forecast used by scheduling policies",
            provider="Open-Meteo",
            request_url=weather_metadata["forecast_url"],
            retrieved_at=retrieved_at,
            license_name="CC BY 4.0",
            attribution="Weather data by Open-Meteo.com",
            provenance="external_derived",
            transformations=[
                "Selected temperature_2m_previous_day1 for a fixed 24-hour lead.",
                "Renamed the variable to air_temperature and normalized timestamps to UTC.",
            ],
            timing=SourceTiming(
                issue_time_basis="provider",
                availability_basis="scenario_assumption",
                note=(
                    "The fixed lead establishes issue time; decision-time availability is a "
                    "declared scenario assumption because the historical payload omits it."
                ),
            ),
            limitations=["Forecast values are gridded model output, not a site sensor."],
        ),
        _record(
            source_id="open-meteo-weather-settlement",
            artifact=actual,
            selection="metric=air_temperature; quality=estimated",
            role="post-horizon weather used for counterfactual settlement",
            provider="Open-Meteo",
            request_url=weather_metadata["actual_url"],
            retrieved_at=retrieved_at,
            license_name="CC BY 4.0",
            attribution="Weather data by Open-Meteo.com",
            provenance="external_derived",
            transformations=["Selected hourly temperature_2m and marked it as an estimated value."],
            timing=SourceTiming(
                issue_time_basis="not_applicable",
                availability_basis="retrieval",
                note="Settlement rows become available at the recorded post-horizon retrieval.",
            ),
            limitations=["Values are gridded model estimates, not measured site telemetry."],
        ),
        _record(
            source_id="neso-carbon-forecast",
            artifact=grid,
            selection="signal=carbon_intensity; quality=forecast",
            role="national GB carbon forecast used by scheduling policies",
            provider="NESO Carbon Intensity API",
            request_url=carbon_metadata["url"],
            retrieved_at=retrieved_at,
            license_name="CC BY 4.0",
            attribution="Carbon intensity data by National Energy System Operator",
            provenance="external_derived",
            transformations=["Averaged two national half-hour values into each hourly value."],
            timing=SourceTiming(
                issue_time_basis="scenario_assumption",
                availability_basis="scenario_assumption",
                note=(
                    "The historical response omits original issue and retrieval timestamps; the "
                    "decision-time freeze is a scenario assumption."
                ),
            ),
            limitations=["The national GB signal does not resolve London regional variation."],
        ),
        _record(
            source_id="neso-carbon-estimated-actual",
            artifact=grid,
            selection="signal=carbon_intensity; quality=estimated",
            role="national GB estimated actual carbon intensity used for settlement",
            provider="NESO Carbon Intensity API",
            request_url=carbon_metadata["url"],
            retrieved_at=retrieved_at,
            license_name="CC BY 4.0",
            attribution="Carbon intensity data by National Energy System Operator",
            provenance="external_derived",
            transformations=["Averaged two national half-hour values into each hourly value."],
            timing=SourceTiming(
                issue_time_basis="not_applicable",
                availability_basis="retrieval",
                note="Estimated actual values are settlement-only at the retrieval timestamp.",
            ),
            limitations=["Provider actual values are estimates, not metered site emissions."],
        ),
        _record(
            source_id="declared-scenario-tariff",
            artifact=grid,
            selection="signal=energy_price; forecast and estimated settlement rows",
            role="deterministic time-of-use price scenario",
            provider="ClimaDC project fixture",
            request_url=_PROJECT_URL,
            retrieved_at=retrieved_at,
            license_name="Apache-2.0",
            attribution="ClimaDC project-owned deterministic scenario",
            provenance="project_generated",
            transformations=[],
            timing=SourceTiming(
                issue_time_basis="scenario_assumption",
                availability_basis="scenario_assumption",
                note="Forecast and settlement tariff values are deliberately identical.",
            ),
            limitations=["The tariff is not a supplier product or observed bill."],
        ),
        _record(
            source_id="deterministic-workload",
            artifact=workload,
            selection="all rows",
            role="deadline-constrained batch workload scenario",
            provider="ClimaDC project fixture",
            request_url=_PROJECT_URL,
            retrieved_at=retrieved_at,
            license_name="Apache-2.0",
            attribution="ClimaDC project-owned deterministic scenario",
            provenance="project_generated",
            transformations=[],
            timing=SourceTiming(
                issue_time_basis="not_applicable",
                availability_basis="scenario_assumption",
                note="All jobs are deliberately available at the replay decision time.",
            ),
            limitations=["Jobs are synthetic and do not represent production traces."],
        ),
    ]
    return SourceManifest(study_id="gb-london-carbon-shift-24h", records=records)


def refresh_carbon_shift(
    destination: Path,
    *,
    decision_time: pd.Timestamp,
    retrieved_at: pd.Timestamp | None = None,
    weather_adapter: OpenMeteoHistoryAdapter | None = None,
    carbon_adapter: NESOCarbonIntensityAdapter | None = None,
) -> Path:
    """Create a new immutable local reference snapshot without overwriting an existing path."""

    destination = Path(destination).resolve()
    if destination.exists():
        raise ConfigurationError(f"Refresh destination already exists: {destination}")
    if (
        not isinstance(decision_time, pd.Timestamp)
        or decision_time.tzinfo is None
        or str(decision_time.tzinfo) != "UTC"
    ):
        raise ConfigurationError("decision_time must be an exact UTC pandas Timestamp")
    retrieval = retrieved_at if retrieved_at is not None else pd.Timestamp.now(tz="UTC")
    if (
        not isinstance(retrieval, pd.Timestamp)
        or pd.isna(retrieval)
        or retrieval.tzinfo is None
        or str(retrieval.tzinfo) != "UTC"
    ):
        raise ConfigurationError("retrieved_at must be an exact UTC pandas Timestamp")
    horizon = pd.Timedelta(hours=24)
    if retrieval < decision_time + horizon:
        raise ConfigurationError("retrieved_at must follow the complete 24-hour replay horizon")

    weather_source = weather_adapter or OpenMeteoHistoryAdapter(clock=lambda: retrieval)
    carbon_source = carbon_adapter or NESOCarbonIntensityAdapter(clock=lambda: retrieval)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".climadc-refresh-", dir=destination.parent))
    try:
        weather = weather_source.fetch(
            latitude=_LATITUDE,
            longitude=_LONGITUDE,
            site_id=_SITE_ID,
            decision_time=decision_time,
            horizon=horizon,
        )
        carbon = carbon_source.fetch(site_id=_SITE_ID, decision_time=decision_time, horizon=horizon)
        grid = GridSignalFrame.from_pandas(
            pd.concat([carbon.to_pandas(), _tariff(decision_time, horizon)], ignore_index=True)
        )
        workload = _workload(decision_time)
        weather.forecast.to_pandas().to_csv(temporary / "climate-forecast.csv", index=False)
        weather.actual.to_pandas().to_csv(temporary / "actual-weather.csv", index=False)
        grid.to_pandas().to_csv(temporary / "grid-signals.csv", index=False)
        workload.to_pandas().to_csv(temporary / "workload.csv", index=False)
        manifest = _manifest(
            temporary,
            retrieved_at=retrieval,
            weather_metadata=dict(weather.metadata),
            carbon_metadata=dict(carbon_source.metadata),
        )
        (temporary / "source-manifest.yaml").write_text(
            yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=True, allow_unicode=False),
            encoding="utf-8",
            newline="\n",
        )
        (temporary / "study.yaml").write_text(
            yaml.safe_dump(_study_payload(decision_time), sort_keys=False, allow_unicode=False),
            encoding="utf-8",
            newline="\n",
        )
        temporary.rename(destination)
        return destination / "study.yaml"
    except ConfigurationError:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    except Exception as exc:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise ConfigurationError(f"Unable to refresh carbon-shift fixture: {exc}") from exc
