"""Regenerate the project-owned CC0 WeatherDC small fixture."""

from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).parent
START = datetime(2026, 1, 1)
PERIODS = 48
FORECAST_ISSUE = "2025-12-30 00:00"


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for hour in range(PERIODS):
        timestamp = START + timedelta(hours=hour)
        phase = 2.0 * math.pi * (hour % 24) / 24.0
        temp = 27.0 + 5.0 * math.sin(phase) + 0.04 * hour
        humid = 61.0 + 9.0 * math.cos(phase)
        press = 1008.0 + 2.0 * math.cos(phase / 2.0)
        rain = 0.6 if hour % 17 == 0 else 0.0
        solar = max(0.0, 720.0 * math.sin(phase - math.pi / 2.0))
        cooling = 12.0 + 2.0 * temp + 0.08 * humid + 0.01 * solar
        it_power = 33.0 + 0.8 * math.cos(phase * 2.0)
        rows.append(
            {
                "timestamp": timestamp,
                "temp": temp,
                "humid": humid,
                "press": press,
                "rain": rain,
                "solar": solar,
                "CRAC3": cooling * 0.54,
                "CRAC4": cooling * 0.46,
                "ULC5": it_power * 0.51,
                "ULC6": it_power * 0.49,
                "cooling_power": cooling,
            }
        )
    return rows


def _write_weather(rows: list[dict[str, object]], variable: str) -> None:
    with (ROOT / f"BST1_{variable}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["year", "month", "day", "time", variable, "forecast_issue_time"])
        for row in rows:
            timestamp = row["timestamp"]
            assert isinstance(timestamp, datetime)
            writer.writerow(
                [
                    timestamp.strftime("%Y"),
                    timestamp.strftime("%m"),
                    timestamp.strftime("%d"),
                    timestamp.strftime("%H:%M"),
                    f"{float(row[variable]):.6f}",
                    FORECAST_ISSUE,
                ]
            )


def _write_meter(rows: list[dict[str, object]], meter: str) -> None:
    with (ROOT / f"powertest_{meter}.out.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Timestamp", "Active_Threephase_Power"])
        for row in rows:
            timestamp = row["timestamp"]
            assert isinstance(timestamp, datetime)
            writer.writerow([timestamp.strftime("%d/%m/%Y %H:%M:%S"), f"{float(row[meter]):.6f}"])


def _write_expected(rows: list[dict[str, object]]) -> None:
    with (ROOT / "expected.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["event_time", "cooling_power"])
        for row in rows:
            timestamp = row["timestamp"]
            assert isinstance(timestamp, datetime)
            utc = timestamp - timedelta(hours=7)
            observed = round(float(row["CRAC3"]), 6) + round(float(row["CRAC4"]), 6)
            writer.writerow([utc.isoformat() + "+00:00", f"{observed:.6f}"])


def _write_workload(rows: list[dict[str, object]]) -> None:
    with (ROOT / "workload.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "job_id",
                "site_id",
                "event_time",
                "available_at",
                "deadline",
                "resource_type",
                "demand",
                "unit",
                "flexible_fraction",
            ]
        )
        for hour, row in enumerate(rows):
            timestamp = row["timestamp"]
            assert isinstance(timestamp, datetime)
            event = timestamp - timedelta(hours=7)
            deadline = event + timedelta(hours=8)
            writer.writerow(
                [
                    f"synthetic-{hour:03d}",
                    "weatherdc-kasetsart",
                    event.isoformat() + "+00:00",
                    event.isoformat() + "+00:00",
                    deadline.isoformat() + "+00:00",
                    "compute_energy",
                    f"{8.0 + (hour % 4):.6f}",
                    "kWh",
                    "0.5",
                ]
            )


def main() -> None:
    rows = _rows()
    for variable in ("temp", "humid", "press", "rain", "solar"):
        _write_weather(rows, variable)
    for meter in ("CRAC3", "CRAC4", "ULC5", "ULC6"):
        _write_meter(rows, meter)
    _write_expected(rows)
    _write_workload(rows)


if __name__ == "__main__":
    main()
