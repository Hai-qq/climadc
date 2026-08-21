from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from climadc import __version__
from climadc.evidence.claims import ClaimRegistry
from examples.weatherdc_kasetsart.run import run_small

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "benchmarks" / "reference" / "gb_london_24h" / "summary.json"


@pytest.mark.parametrize(
    "relative_path",
    [
        "benchmarks/weatherdc.yaml",
        "benchmarks/reference/gb_london_24h/summary.json",
        "benchmarks/reference/gb_london_24h/summary.csv",
    ],
)
def test_byte_bound_repository_evidence_uses_lf(relative_path: str) -> None:
    payload = (ROOT / relative_path).read_bytes()

    assert payload.endswith(b"\n")
    assert b"\r\n" not in payload


def test_claim_registry_is_strict_and_london_claim_matches_golden() -> None:
    registry = ClaimRegistry.from_yaml(ROOT / "evidence" / "claims.yaml")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    claim = next(item for item in registry.claims if item.claim_id == "E1-LONDON-TRADEOFF-001")
    price = next(item for item in golden["policies"] if item["policy"] == "price")

    assert claim.evidence_level == "E1"
    assert claim.status == "illustrative"
    assert claim.scenario_or_study_id == golden["study_id"]
    assert claim.config_sha256 == golden["config_sha256"]
    assert claim.input_hashes == golden["input_hashes"]
    assert claim.output_sha256 == hashlib.sha256(GOLDEN.read_bytes()).hexdigest()
    assert price["energy_cost_change_vs_asap"] == pytest.approx(-28.3185)
    assert price["estimated_location_based_emissions_change_vs_asap_kgco2e"] == pytest.approx(
        41.78469375
    )
    assert "operational savings" in " ".join(claim.limitations).lower()


def test_weatherdc_claim_binds_the_generated_metrics_file(tmp_path: Path) -> None:
    registry = ClaimRegistry.from_yaml(ROOT / "evidence" / "claims.yaml")
    claim = next(item for item in registry.claims if item.claim_id == "E0-WEATHERDC-SANITY-001")
    result, run_dir = run_small(tmp_path / "weatherdc")

    assert (
        claim.config_sha256
        == hashlib.sha256((ROOT / "benchmarks/weatherdc.yaml").read_bytes()).hexdigest()
    )
    assert claim.input_hashes == result.input_hashes
    assert (
        claim.output_sha256 == hashlib.sha256((run_dir / "metrics.json").read_bytes()).hexdigest()
    )


def test_golden_summary_is_generated_and_current() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "reference" / "gb_london_24h" / "reproduce.py"),
            "--check",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert payload["climadc_version"] == __version__
    assert payload["interpretation"] == (
        "single-day mechanism demonstration; not a production savings claim"
    )
    assert payload["comparison_tolerance"]["rationale"]


def test_golden_comparators_enforce_declared_tolerance() -> None:
    namespace = runpy.run_path(
        str(ROOT / "benchmarks" / "reference" / "gb_london_24h" / "reproduce.py")
    )
    json_equivalent = namespace["_json_equivalent"]
    csv_equivalent = namespace["_csv_equivalent"]
    first_difference = namespace["_first_difference"]

    assert json_equivalent('{"metric": 1.0}', '{"metric": 1.000000005}')
    assert not json_equivalent('{"metric": 1.0}', '{"metric": 1.00000002}')
    assert not json_equivalent('{"metric": 1.0}', '{"different": 1.0}')
    assert csv_equivalent("policy,metric\nasap,1.0\n", "policy,metric\nasap,1.000000005\n")
    assert not csv_equivalent("policy,metric\nasap,1.0\n", "policy,metric\nasap,1.00000002\n")
    assert first_difference({"metric": 1.0}, {"metric": 1.00000002}).startswith(
        "$.metric: expected 1.0, generated 1.00000002, absolute difference "
    )
