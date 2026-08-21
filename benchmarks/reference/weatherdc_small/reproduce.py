from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from climadc import __version__  # noqa: E402
from examples.weatherdc_kasetsart.run import run_small  # noqa: E402

HERE = Path(__file__).resolve().parent
SUMMARY_JSON = HERE / "summary.json"
ABSOLUTE_TOLERANCE = 1e-12


def generated_payload() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="climadc-weatherdc-reference-") as temporary:
        result, _ = run_small(Path(temporary) / "run")
    return {
        "schema_version": "1",
        "study_id": result.study_id,
        "evidence_level": "E0",
        "interpretation": "project-generated synthetic pipeline sanity check",
        "climadc_version": __version__,
        "config_sha256": hashlib.sha256(
            (ROOT / "benchmarks/weatherdc.yaml").read_bytes()
        ).hexdigest(),
        "input_hashes": dict(sorted(result.input_hashes.items())),
        "comparison_tolerance": {
            "absolute": ABSOLUTE_TOLERANCE,
            "relative": 0.0,
            "rationale": (
                "The WeatherDC OLS path is compared after serialization; 1e-12 absorbs "
                "cross-platform LAPACK rounding while remaining below half the least precise "
                "decimal place published by the E0 claim."
            ),
        },
        "metrics": result.metrics["cooling_power"],
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the compact WeatherDC small E0 summary"
    )
    parser.add_argument("--check", action="store_true", help="compare without modifying files")
    args = parser.parse_args()
    generated = generated_payload()
    if args.check:
        if not SUMMARY_JSON.is_file():
            parser.error(f"golden summary is missing: {SUMMARY_JSON}")
        expected = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        difference = _first_difference(expected, generated)
        if difference is not None:
            parser.error(f"golden summary differs: {SUMMARY_JSON} ({difference})")
        return 0
    SUMMARY_JSON.write_text(
        json.dumps(generated, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
