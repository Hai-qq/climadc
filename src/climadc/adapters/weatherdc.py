from __future__ import annotations

import hashlib
import os
import re
import stat
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import cast
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from climadc.contracts.frames import ClimateForecastFrame, DCTelemetryFrame
from climadc.errors import ConfigurationError

_SITE_ID = "weatherdc-kasetsart"
_TIMEZONE = "Asia/Bangkok"
_METER_NAMES = ("CRAC3", "CRAC4", "ULC5", "ULC6")
_WEATHER_UNITS = {
    "temp": "degC",
    "humid": "percent",
    "press": "hPa",
    "rain": "mm",
    "solar": "W / m^2",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RETRIES = 3
_TIMEOUT_SECONDS = 60.0
_USER_AGENT = "climadc/0.1 WeatherDC adapter"

Downloader = Callable[[str, Path], None]
Sleeper = Callable[[float], None]


@dataclass(frozen=True)
class SourceItem:
    name: str
    url: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class SourceManifest:
    items: tuple[SourceItem, ...]


@dataclass(frozen=True)
class DownloadRecord:
    name: str
    path: Path
    sha256: str
    bytes: int


@dataclass(frozen=True)
class DownloadManifest:
    records: tuple[DownloadRecord, ...]


WEATHERDC_SOURCE_MANIFEST = SourceManifest(
    items=(
        SourceItem(
            "powertest_CRAC3.out.csv",
            "https://github.com/cchantra/energydata/raw/refs/heads/master/examples/"
            "powertest_CRAC3.out.csv",
            "faae74d9ff4e60a10d8c566173d22a33231465f896a6b41f5a2988726ef5343e",
            31_364_327,
        ),
        SourceItem(
            "powertest_CRAC4.out.csv",
            "https://github.com/cchantra/energydata/raw/refs/heads/master/examples/"
            "powertest_CRAC4.out.csv",
            "a1d410867b4dee5304696577c4d191c67fe2272cc16908fb8f864fcd24989d62",
            31_488_954,
        ),
        SourceItem(
            "powertest_ULC5.out.csv",
            "https://github.com/cchantra/energydata/raw/refs/heads/master/examples/"
            "powertest_ULC5.out.csv",
            "61bd063f2684e0b4819ac8f81769604d6c19bd832739878dd6909d19f167b55b",
            32_490_219,
        ),
        SourceItem(
            "powertest_ULC6.out.csv",
            "https://github.com/cchantra/energydata/raw/refs/heads/master/examples/"
            "powertest_ULC6.out.csv",
            "414b42bf5414426c5287cce1254b7aefe3c302eb594cdac5d7f096b3118c181b",
            32_385_783,
        ),
        SourceItem(
            "BST1_temp.csv",
            "https://tiservice.hii.or.th/opendata/backup_catalog/weather/clean/"
            "weather10year2009-2018/temp/BST1.csv",
            "564d10305c552407cf1cdcec73e3a13bc0d3b9896ccd82cd185ecb95c34f1c8a",
            1_912_321,
        ),
        SourceItem(
            "BST1_humid.csv",
            "https://tiservice.hii.or.th/opendata/backup_catalog/weather/clean/"
            "weather10year2009-2018/humid/BST1.csv",
            "69abfa60775ead0d215254ae335ba6f7d45948babb7dc2af53298ce023650f21",
            1_828_231,
        ),
        SourceItem(
            "BST1_press.csv",
            "https://tiservice.hii.or.th/opendata/backup_catalog/weather/clean/"
            "weather10year2009-2018/press/BST1.csv",
            "ac56f5580f44c1bad4db0ff6b689683f9747b74b2d08da5c6f33c9fd5e409a7a",
            1_933_044,
        ),
        SourceItem(
            "BST1_rain.csv",
            "https://tiservice.hii.or.th/opendata/backup_catalog/weather/clean/"
            "weather10year2009-2018/rain/BST1.csv",
            "4924fe4fd661c77c18572491982d1f36df18c8afdb9a51dd393e56ec7a776fd7",
            1_698_417,
        ),
        SourceItem(
            "BST1_solar.csv",
            "https://tiservice.hii.or.th/opendata/backup_catalog/weather/clean/"
            "weather10year2009-2018/solar/BST1.csv",
            "c7a67402cec442664f6fffd348dbaf1147907b8b3b12fba691ea77a13fdb930b",
            1_785_140,
        ),
        SourceItem(
            "0metadata_weather.csv",
            "https://tiservice.hii.or.th/opendata/backup_catalog/weather/clean/"
            "weather10year2009-2018/0metadata_weather.csv",
            "c60c67c5c3576286cf809ed79f3ae45dbae5e5944f3ac07374695aa2caf671a9",
            197_890,
        ),
    )
)


def _default_downloader(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        with destination.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)


def _direct_filename(name: object) -> str:
    windows = PureWindowsPath(str(name))
    if (
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or Path(name).name != name
        or "/" in name
        or "\\" in name
        or name.endswith(".part")
        or name in {".", ".."}
    ):
        raise ConfigurationError("WeatherDC source name must be a safe direct filename")
    return name


def _validate_item(item: SourceItem) -> None:
    _direct_filename(item.name)
    parsed = urlsplit(item.url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError("WeatherDC source URL must be absolute HTTP(S)")
    if not _SHA256.fullmatch(item.sha256):
        raise ConfigurationError("WeatherDC source sha256 must be 64 lowercase hex characters")
    if not isinstance(item.bytes, int) or isinstance(item.bytes, bool) or item.bytes < 0:
        raise ConfigurationError("WeatherDC source bytes must be a nonnegative integer")


def _regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, item: SourceItem) -> bool:
    return (
        _regular_file(path)
        and path.stat().st_size == item.bytes
        and _file_sha256(path) == item.sha256
    )


def _local_times(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if bool(parsed.isna().any()):
        raise ConfigurationError("WeatherDC contains an invalid local timestamp")
    try:
        localized = parsed.dt.tz_localize(_TIMEZONE, ambiguous="raise", nonexistent="raise")
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("WeatherDC timestamp cannot be localized") from exc
    return cast(pd.Series, localized.dt.tz_convert("UTC"))


def _load_climate(cache_dir: Path) -> ClimateForecastFrame:
    rows: list[pd.DataFrame] = []
    for variable, unit in _WEATHER_UNITS.items():
        path = cache_dir / f"BST1_{variable}.csv"
        try:
            raw = pd.read_csv(
                path,
                dtype={"year": str, "month": str, "day": str, "time": str},
            )
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise ConfigurationError(f"Unable to read WeatherDC climate source {path}") from exc
        required = {"year", "month", "day", "time", variable}
        if not required.issubset(raw.columns):
            raise ConfigurationError(f"WeatherDC climate source {path} has an invalid schema")
        local = raw["year"] + "-" + raw["month"] + "-" + raw["day"] + " " + raw["time"]
        valid_time = _local_times(local)
        values = pd.to_numeric(raw[variable], errors="coerce")
        keep = values.notna() & (values != -999)
        if variable == "humid":
            keep &= values.between(0, 100)
        if variable in {"rain", "solar"}:
            keep &= values >= 0
        if not bool(keep.any()):
            raise ConfigurationError(f"WeatherDC climate source {path} has no valid values")
        if "forecast_issue_time" in raw.columns:
            issue_time = _local_times(raw["forecast_issue_time"])
            source = "weatherdc:synthetic-forecast"
        else:
            issue_time = valid_time.copy()
            source = "weatherdc:hii-observation"
        selected = pd.DataFrame(
            {
                "site_id": _SITE_ID,
                "issue_time": issue_time.loc[keep].reset_index(drop=True),
                "available_at": issue_time.loc[keep].reset_index(drop=True),
                "valid_time": valid_time.loc[keep].reset_index(drop=True),
                "variable": variable,
                "value": values.loc[keep].astype(float).reset_index(drop=True),
                "unit": unit,
                "source": source,
                "quantile": pd.NA,
                "member": pd.NA,
            }
        )
        rows.append(selected)
    return ClimateForecastFrame.from_pandas(pd.concat(rows, ignore_index=True))


def _load_meter_series(cache_dir: Path) -> dict[str, pd.Series]:
    meters: dict[str, pd.Series] = {}
    for meter in _METER_NAMES:
        path = cache_dir / f"powertest_{meter}.out.csv"
        try:
            raw = pd.read_csv(path, usecols=["Timestamp", "Active_Threephase_Power"])
        except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
            raise ConfigurationError(f"Unable to read WeatherDC meter source {path}") from exc
        parsed = pd.to_datetime(raw["Timestamp"], dayfirst=True, errors="coerce")
        values = pd.to_numeric(raw["Active_Threephase_Power"], errors="coerce")
        valid = parsed.notna() & values.notna() & np.isfinite(values) & (values >= 0)
        if not bool(valid.any()):
            raise ConfigurationError(f"WeatherDC meter source {path} has no valid values")
        local_index = pd.DatetimeIndex(parsed.loc[valid]).tz_localize(
            _TIMEZONE, ambiguous="raise", nonexistent="raise"
        )
        series = pd.Series(
            values.loc[valid].to_numpy(dtype=float), index=local_index.tz_convert("UTC")
        )
        meters[meter] = series.sort_index().resample("1h").median().dropna()
    return meters


def _load_telemetry(cache_dir: Path) -> DCTelemetryFrame:
    meters = _load_meter_series(cache_dir)
    aligned = pd.concat(meters, axis=1, join="inner").dropna()
    if aligned.empty:
        raise ConfigurationError("WeatherDC meter sources have no shared hourly timestamps")
    rows: list[dict[str, object]] = []
    for timestamp, values in aligned.iterrows():
        for meter in _METER_NAMES:
            rows.append(
                {
                    "site_id": _SITE_ID,
                    "device_id": meter,
                    "event_time": timestamp,
                    "available_at": timestamp,
                    "metric": "meter_power",
                    "value": float(values[meter]),
                    "unit": "kW",
                    "quality": "observed",
                }
            )
        aggregates = {
            "cooling_power": float(values["CRAC3"] + values["CRAC4"]),
            "it_power": float(values["ULC5"] + values["ULC6"]),
        }
        aggregates["total_power"] = aggregates["cooling_power"] + aggregates["it_power"]
        for metric, value in aggregates.items():
            rows.append(
                {
                    "site_id": _SITE_ID,
                    "device_id": "aggregate",
                    "event_time": timestamp,
                    "available_at": timestamp,
                    "metric": metric,
                    "value": value,
                    "unit": "kW",
                    "quality": "observed",
                }
            )
    return DCTelemetryFrame.from_pandas(pd.DataFrame(rows))


class WeatherDCAdapter:
    def __init__(
        self,
        downloader: Downloader | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        self._downloader = downloader if downloader is not None else _default_downloader
        self._sleeper = sleeper if sleeper is not None else time.sleep

    def download(self, cache_dir: Path, manifest: SourceManifest) -> DownloadManifest:
        cache_dir = Path(cache_dir)
        names = [item.name for item in manifest.items]
        if len(names) != len(set(names)):
            raise ConfigurationError("WeatherDC source names must be unique")
        records: list[DownloadRecord] = []
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigurationError(f"Unable to create WeatherDC cache {cache_dir}") from exc
        for item in manifest.items:
            _validate_item(item)
            destination = cache_dir / item.name
            part = cache_dir / f"{item.name}.part"
            if _verified(destination, item):
                records.append(DownloadRecord(item.name, destination, item.sha256, item.bytes))
                continue
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            last_error: Exception | None = None
            for attempt in range(_RETRIES):
                try:
                    if part.exists() or part.is_symlink():
                        part.unlink()
                    self._downloader(item.url, part)
                    if not _verified(part, item):
                        raise ConfigurationError(
                            "downloaded bytes or SHA-256 do not match manifest"
                        )
                    os.replace(part, destination)
                    last_error = None
                    break
                except Exception as exc:  # downloader implementations define their own errors
                    last_error = exc
                    if part.exists() or part.is_symlink():
                        part.unlink()
                    if attempt + 1 < _RETRIES:
                        self._sleeper(float(2**attempt))
            if last_error is not None:
                raise ConfigurationError(
                    f"WeatherDC source {item.name} failed verification after 3 attempts"
                ) from last_error
            records.append(DownloadRecord(item.name, destination, item.sha256, item.bytes))
        return DownloadManifest(records=tuple(records))

    def load(self, cache_dir: Path) -> tuple[ClimateForecastFrame, DCTelemetryFrame]:
        cache_dir = Path(cache_dir)
        return _load_climate(cache_dir), _load_telemetry(cache_dir)
