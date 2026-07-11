from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from climadc.adapters.weatherdc import SourceItem, SourceManifest, WeatherDCAdapter
from climadc.errors import ConfigurationError


FIXTURE = Path(__file__).parents[1] / "fixtures" / "weatherdc_small"


def _item(payload: bytes, name: str = "source.csv") -> SourceItem:
    return SourceItem(
        name=name,
        url="https://data.example.org/source.csv",
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )


def test_download_retries_to_verified_part_then_atomically_publishes(tmp_path: Path) -> None:
    payload = b"verified WeatherDC source\n"
    attempts: list[tuple[str, Path]] = []

    def downloader(url: str, destination: Path) -> None:
        attempts.append((url, destination))
        if len(attempts) < 3:
            raise OSError("transient")
        destination.write_bytes(payload)

    manifest = WeatherDCAdapter(downloader=downloader, sleeper=lambda _: None).download(
        tmp_path, SourceManifest(items=(_item(payload),))
    )

    assert len(attempts) == 3
    assert all(url == "https://data.example.org/source.csv" for url, _ in attempts)
    assert all(path == tmp_path / "source.csv.part" for _, path in attempts)
    assert manifest.records[0].path == tmp_path / "source.csv"
    assert manifest.records[0].path.read_bytes() == payload
    assert not (tmp_path / "source.csv.part").exists()


def test_download_reuses_only_a_byte_and_hash_verified_cache(tmp_path: Path) -> None:
    payload = b"cached\n"
    destination = tmp_path / "source.csv"
    destination.write_bytes(payload)

    def unexpected_download(url: str, path: Path) -> None:
        raise AssertionError(f"unexpected download {url} to {path}")

    result = WeatherDCAdapter(downloader=unexpected_download).download(
        tmp_path, SourceManifest(items=(_item(payload),))
    )

    assert result.records[0].path == destination


def test_download_rejects_corrupt_payload_after_exactly_three_attempts(tmp_path: Path) -> None:
    attempts = 0

    def corrupt_download(url: str, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        destination.write_bytes(b"wrong")

    with pytest.raises(ConfigurationError, match="after 3 attempts"):
        WeatherDCAdapter(downloader=corrupt_download, sleeper=lambda _: None).download(
            tmp_path, SourceManifest(items=(_item(b"expected"),))
        )

    assert attempts == 3
    assert not (tmp_path / "source.csv").exists()
    assert not (tmp_path / "source.csv.part").exists()


@pytest.mark.parametrize("name", ["../escape.csv", "/tmp/escape.csv", "nested/file.csv", "x.part"])
def test_download_rejects_unsafe_source_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ConfigurationError, match="safe direct filename"):
        WeatherDCAdapter().download(tmp_path, SourceManifest(items=(_item(b"x", name=name),)))


def test_load_converts_four_meters_and_five_climate_variables_to_utc() -> None:
    climate, telemetry = WeatherDCAdapter().load(FIXTURE)
    climate_rows = climate.to_pandas()
    telemetry_rows = telemetry.to_pandas()

    assert set(climate_rows["variable"]) == {"temp", "humid", "press", "rain", "solar"}
    assert set(telemetry_rows["device_id"].dropna()) >= {"CRAC3", "CRAC4", "ULC5", "ULC6"}
    assert {"cooling_power", "it_power", "total_power"}.issubset(set(telemetry_rows["metric"]))
    assert str(climate_rows["valid_time"].dtype) == "datetime64[ns, UTC]"
    assert str(telemetry_rows["event_time"].dtype) == "datetime64[ns, UTC]"
    assert climate_rows["valid_time"].min() == pd.Timestamp("2025-12-31 17:00:00+00:00")
    assert telemetry_rows["event_time"].min() == pd.Timestamp("2025-12-31 17:00:00+00:00")
    assert (climate_rows["issue_time"] == climate_rows["available_at"]).all()
    assert (climate_rows["available_at"] <= climate_rows["valid_time"]).all()
    assert (telemetry_rows["available_at"] == telemetry_rows["event_time"]).all()
