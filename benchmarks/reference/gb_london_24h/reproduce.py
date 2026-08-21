from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from typing import Any, cast

import pandas as pd

from climadc import __version__
from climadc.reference import packaged_study_path
from climadc.replay import ReplayStudyConfig, ReplayStudyRunner

HERE = Path(__file__).resolve().parent
SUMMARY_JSON = HERE / "summary.json"
SUMMARY_CSV = HERE / "summary.csv"
ABSOLUTE_TOLERANCE = 1e-8


def _plain(value: object) -> object:
    if hasattr(value, "item"):
        return cast(Any, value).item()
    return value


def generated_text() -> tuple[str, str]:
    config = ReplayStudyConfig.from_yaml(packaged_study_path())
    result = ReplayStudyRunner(clock=lambda: pd.Timestamp("2026-08-21T00:00:00Z")).run(config)
    records = [
        {str(key): _plain(value) for key, value in row.items()}
        for row in result.replay.metrics.to_dict(orient="records")
    ]
    payload = {
        "schema_version": "1",
        "study_id": config.study_id,
        "evidence_level": "E1",
        "interpretation": "single-day mechanism demonstration; not a production savings claim",
        "climadc_version": __version__,
        "config_sha256": result.config_sha256,
        "input_hashes": dict(sorted(result.input_hashes.items())),
        "objective": config.replay.objective_payload(),
        "units": {
            "facility_energy_kwh": "kWh",
            "it_energy_kwh": "kWh",
            "cooling_energy_kwh": "kWh",
            "estimated_location_based_emissions_kgco2e": "kgCO2e",
            "energy_charge": result.replay.currency,
            "demand_charge": result.replay.currency,
            "energy_cost": result.replay.currency,
            "peak_kw": "kW",
            "shifted_energy_kwh": "kWh",
            "energy_cost_change_vs_asap": result.replay.currency,
            "estimated_location_based_emissions_change_vs_asap_kgco2e": "kgCO2e",
            "peak_change_vs_asap_kw": "kW",
            "realized_objective": result.replay.currency,
            "objective_regret": result.replay.currency,
        },
        "comparison_tolerance": {
            "absolute": ABSOLUTE_TOLERANCE,
            "relative": 0.0,
            "rationale": (
                "The deterministic linear program is compared after serialization; 1e-8 absorbs "
                "cross-platform floating-point solver noise while remaining below the configured "
                "1e-7 kWh feasibility tolerance."
            ),
        },
        "policies": records,
    }
    json_text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return json_text, stream.getvalue()


def _equivalent(expected: object, actual: object) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            _equivalent(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _equivalent(expected_item, actual_item)
            for expected_item, actual_item in zip(expected, actual, strict=True)
        )
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        if isinstance(expected, int) and isinstance(actual, int):
            return expected == actual
        return math.isclose(
            float(expected),
            float(actual),
            rel_tol=0.0,
            abs_tol=ABSOLUTE_TOLERANCE,
        )
    return type(expected) is type(actual) and expected == actual


def _first_difference(expected: object, actual: object, path: str = "$") -> str | None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return f"{path}: keys differ"
        for key in expected:
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path}: lengths differ ({len(expected)} != {len(actual)})"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            difference = _first_difference(expected_item, actual_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        if _equivalent(expected, actual):
            return None
        return (
            f"{path}: expected {expected!r}, generated {actual!r}, "
            f"absolute difference {abs(float(expected) - float(actual))!r}"
        )
    if type(expected) is type(actual) and expected == actual:
        return None
    return f"{path}: expected {expected!r}, generated {actual!r}"


def _json_equivalent(expected_text: str, actual_text: str) -> bool:
    try:
        return _equivalent(json.loads(expected_text), json.loads(actual_text))
    except (TypeError, ValueError):
        return False


def _csv_equivalent(expected_text: str, actual_text: str) -> bool:
    try:
        expected_reader = csv.DictReader(io.StringIO(expected_text))
        actual_reader = csv.DictReader(io.StringIO(actual_text))
        if expected_reader.fieldnames != actual_reader.fieldnames:
            return False
        expected_rows = list(expected_reader)
        actual_rows = list(actual_reader)
    except (csv.Error, TypeError, ValueError):
        return False
    if len(expected_rows) != len(actual_rows):
        return False
    for expected_row, actual_row in zip(expected_rows, actual_rows, strict=True):
        if expected_row.keys() != actual_row.keys():
            return False
        for key, expected in expected_row.items():
            actual = actual_row[key]
            if expected == actual:
                continue
            try:
                if math.isclose(
                    float(expected),
                    float(actual),
                    rel_tol=0.0,
                    abs_tol=ABSOLUTE_TOLERANCE,
                ):
                    continue
            except (TypeError, ValueError):
                pass
            return False
    return True


def _difference_detail(path: Path, expected_text: str, actual_text: str) -> str:
    if path.suffix == ".json":
        try:
            difference = _first_difference(json.loads(expected_text), json.loads(actual_text))
        except (TypeError, ValueError):
            return "invalid JSON"
        return difference or "serialization differs"
    try:
        expected_reader = csv.DictReader(io.StringIO(expected_text))
        actual_reader = csv.DictReader(io.StringIO(actual_text))
        expected_rows = list(expected_reader)
        actual_rows = list(actual_reader)
    except (csv.Error, TypeError, ValueError):
        return "invalid CSV"
    if expected_reader.fieldnames != actual_reader.fieldnames:
        return "CSV columns differ"
    if len(expected_rows) != len(actual_rows):
        return f"CSV row counts differ ({len(expected_rows)} != {len(actual_rows)})"
    for row_index, (expected_row, actual_row) in enumerate(
        zip(expected_rows, actual_rows, strict=True)
    ):
        for key, expected in expected_row.items():
            actual = actual_row[key]
            if expected == actual:
                continue
            try:
                expected_number = float(expected)
                actual_number = float(actual)
            except (TypeError, ValueError):
                return f"row {row_index}, column {key}: expected {expected!r}, generated {actual!r}"
            if not math.isclose(
                expected_number,
                actual_number,
                rel_tol=0.0,
                abs_tol=ABSOLUTE_TOLERANCE,
            ):
                return (
                    f"row {row_index}, column {key}: expected {expected_number!r}, "
                    f"generated {actual_number!r}, absolute difference "
                    f"{abs(expected_number - actual_number)!r}"
                )
    return "serialization differs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the compact London E1 golden summary")
    parser.add_argument("--check", action="store_true", help="compare without modifying files")
    args = parser.parse_args()
    json_text, csv_text = generated_text()
    if args.check:
        comparisons = (
            (SUMMARY_JSON, json_text, _json_equivalent),
            (SUMMARY_CSV, csv_text, _csv_equivalent),
        )
        mismatches = []
        for path, actual_text, comparator in comparisons:
            if not path.is_file():
                mismatches.append(f"{path} (missing)")
                continue
            expected_text = path.read_text(encoding="utf-8")
            if not comparator(expected_text, actual_text):
                detail = _difference_detail(path, expected_text, actual_text)
                mismatches.append(f"{path} ({detail})")
        if mismatches:
            parser.error("golden summary differs: " + ", ".join(mismatches))
        return 0
    SUMMARY_JSON.write_text(json_text, encoding="utf-8", newline="\n")
    SUMMARY_CSV.write_text(csv_text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
