from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from climadc.adapters.google_clusterdata import (
    GOOGLE_V3_ARTIFACTS,
    GOOGLE_V3_EXPORT_COLUMNS,
    GoogleV3ConversionConfig,
    convert_google_v3_export,
    verify_google_v3_conversion,
)
from climadc.cli.app import app
from climadc.errors import ConfigurationError
from climadc.evidence.checksums import write_checksums

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _source_rows() -> list[dict[str, str]]:
    return [
        {
            "collection_id": "101",
            "instance_index": "0",
            "submit_time_us": "1000000",
            "finish_time_us": "3601000000",
            "requested_cpu": "0.5",
            "priority": "110",
            "scheduling_class": "1",
            "missing_type": "0",
            "collection_type": "0",
            "alloc_collection_id": "0",
            "submit_count": "1",
            "finish_count": "1",
        },
        {
            "collection_id": "100",
            "instance_index": "2",
            "submit_time_us": "2000000",
            "finish_time_us": "1802000000",
            "requested_cpu": "0.25",
            "priority": "50",
            "scheduling_class": "0",
            "missing_type": "0",
            "collection_type": "0",
            "alloc_collection_id": "0",
            "submit_count": "1",
            "finish_count": "1",
        },
    ]


def _write_source(path: Path, rows: list[dict[str, str]] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=GOOGLE_V3_EXPORT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_source_rows() if rows is None else rows)


def _write_query(path: Path, *, cell: str = "a") -> None:
    path.write_text(
        f"SELECT * FROM `google.com:google-cluster-data`.clusterdata_2019_{cell}.instance_events "
        "WHERE time >= @start_time_us AND time < @end_time_us "
        "AND time < @finish_cutoff_time_us\n",
        encoding="utf-8",
        newline="\n",
    )


def _config_payload(input_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "1",
        "dataset": "google-clusterdata-2019-v3",
        "cell": "a",
        "site_id": "google-borg-cell-a-scenario",
        "exported_at": "2026-08-24T00:00:00Z",
        "input_sha256": input_sha256,
        "scenario_epoch": "2030-01-01T00:00:00Z",
        "trace_window": {
            "start_time_us": 0,
            "end_time_us": 4_000_000_000,
            "finish_cutoff_time_us": 8_000_000_000,
        },
        "allowed_scheduling_classes": [0, 1],
        "power_mapping": {
            "kind": "requested_cpu_linear",
            "kw_per_normalized_cpu": 100.0,
            "utilization_fraction": 0.5,
        },
        "deadline_mapping": {
            "kind": "observed_runtime_multiplier",
            "multiplier": 2.0,
        },
        "preemptibility_mapping": {
            "kind": "assume_preemptible",
            "value": True,
        },
    }


def _write_config(path: Path, source: Path, **updates: object) -> None:
    payload = _config_payload(hashlib.sha256(source.read_bytes()).hexdigest())
    payload.update(updates)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
        newline="\n",
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "google-v3.csv"
    config = tmp_path / "conversion.yaml"
    query = tmp_path / "export.sql"
    _write_source(source)
    _write_config(config, source)
    _write_query(query)
    return source, config, query


def test_google_v3_conversion_is_hash_bound_and_reproducible(tmp_path: Path) -> None:
    source, config, query = _inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    result = convert_google_v3_export(source, config, query, first)
    repeated = convert_google_v3_export(source, config, query, second)

    assert result.rows == 2
    assert result.input_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result.workload_sha256 == repeated.workload_sha256
    assert sorted(path.name for path in first.iterdir()) == list(GOOGLE_V3_ARTIFACTS)
    assert (first / "workload.csv").read_bytes() == (second / "workload.csv").read_bytes()

    workload = pd.read_csv(first / "workload.csv")
    assert workload["job_id"].tolist() == ["google-v3-a-100-2", "google-v3-a-101-0"]
    first_job = workload.iloc[0]
    assert first_job["max_power"] == pytest.approx(25.0)
    assert first_job["energy"] == pytest.approx(6.25)
    assert pd.Timestamp(first_job["available_at"]) == pd.Timestamp(first_job["release_time"])
    assert pd.Timestamp(first_job["deadline"]) - pd.Timestamp(
        first_job["release_time"]
    ) == pd.Timedelta(hours=1)

    manifest = json.loads((first / "conversion-manifest.json").read_text(encoding="utf-8"))
    assert manifest["evidence_status"] == "DATA_REQUIRED"
    assert manifest["claim_eligible"] is False
    assert manifest["time_semantics"]["observed_runtime_uses_future_trace_fact"] is True
    assert manifest["source"]["rows"] == 2
    assert manifest["source"]["query_sha256"] == hashlib.sha256(query.read_bytes()).hexdigest()

    offline = verify_google_v3_conversion(first)
    reproduced = verify_google_v3_conversion(first, source_csv=source)
    assert offline.rows == 2 and not offline.source_verified
    assert reproduced.rows == 2 and reproduced.source_verified


def test_google_v3_conversion_refuses_hash_query_and_output_drift(tmp_path: Path) -> None:
    source, config, query = _inputs(tmp_path)
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="SHA-256 mismatch"):
        convert_google_v3_export(source, config, query, tmp_path / "hash-mismatch")

    _write_source(source)
    _write_config(config, source)
    _write_query(query, cell="b")
    with pytest.raises(ConfigurationError, match="declared cell table"):
        convert_google_v3_export(source, config, query, tmp_path / "query-mismatch")

    query.write_text(
        "SELECT * FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_events\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="bounded-window parameters"):
        convert_google_v3_export(source, config, query, tmp_path / "unbounded-query")

    _write_query(query)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(ConfigurationError, match="already exists"):
        convert_google_v3_export(source, config, query, occupied)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("collection_id", "-1", "nonnegative integer"),
        ("missing_type", "1", "missing event data"),
        ("scheduling_class", "2", "outside declared classes"),
        ("submit_count", "2", "exactly one submit"),
        ("finish_count", "0", "exactly one submit"),
        ("collection_type", "1", "not a top-level job task"),
        ("alloc_collection_id", "7", "not a top-level job task"),
        ("requested_cpu", "not-a-number", "must be numeric"),
        ("requested_cpu", "NaN", "finite and positive"),
        ("submit_time_us", "5000000000", "outside trace_window"),
        ("finish_time_us", "1000000", "submit_time_us < finish_time_us"),
    ],
)
def test_google_v3_conversion_rejects_ambiguous_or_unsupported_rows(
    tmp_path: Path,
    column: str,
    value: str,
    message: str,
) -> None:
    rows = _source_rows()
    rows[0][column] = value
    source = tmp_path / "source.csv"
    config = tmp_path / "config.yaml"
    query = tmp_path / "query.sql"
    _write_source(source, rows)
    _write_config(config, source)
    _write_query(query)

    with pytest.raises(ConfigurationError, match=message):
        convert_google_v3_export(source, config, query, tmp_path / "output")


def test_google_v3_conversion_requires_exact_source_schema(tmp_path: Path) -> None:
    source, config, query = _inputs(tmp_path)
    raw = pd.read_csv(source)
    raw["unexpected"] = "value"
    raw.to_csv(source, index=False, lineterminator="\n")
    _write_config(config, source)

    with pytest.raises(ConfigurationError, match="exact columns"):
        convert_google_v3_export(source, config, query, tmp_path / "output")


def test_google_v3_conversion_rejects_empty_duplicate_and_overflowing_tasks(
    tmp_path: Path,
) -> None:
    source, config, query = _inputs(tmp_path)
    _write_source(source, [])
    _write_config(config, source)
    with pytest.raises(ConfigurationError, match="at least one task"):
        convert_google_v3_export(source, config, query, tmp_path / "empty")

    duplicate = _source_rows()
    duplicate.append(dict(duplicate[0]))
    _write_source(source, duplicate)
    _write_config(config, source)
    with pytest.raises(ConfigurationError, match="duplicate task key"):
        convert_google_v3_export(source, config, query, tmp_path / "duplicate")

    overflowing = _source_rows()
    overflowing[0]["requested_cpu"] = "1e309"
    _write_source(source, overflowing)
    _write_config(config, source)
    with pytest.raises(ConfigurationError, match="mapping is non-finite"):
        convert_google_v3_export(source, config, query, tmp_path / "overflow")


def test_google_v3_config_rejects_non_utc_unsorted_and_unknown_fields(tmp_path: Path) -> None:
    source, config, _ = _inputs(tmp_path)
    for updates, message in (
        ({"scenario_epoch": "2030-01-01T00:00:00"}, "timezone-aware UTC"),
        ({"allowed_scheduling_classes": [1, 0]}, "unique and sorted"),
        ({"unknown": True}, "extra_forbidden"),
    ):
        _write_config(config, source, **updates)
        with pytest.raises(ConfigurationError, match=message):
            GoogleV3ConversionConfig.from_yaml(config)


def test_google_v3_config_rejects_invalid_windows_and_nonfinite_mappings(tmp_path: Path) -> None:
    source, config, _ = _inputs(tmp_path)
    invalid_cases = (
        (
            {
                "trace_window": {
                    "start_time_us": 10,
                    "end_time_us": 10,
                    "finish_cutoff_time_us": 20,
                }
            },
            "start_time_us < end_time_us",
        ),
        (
            {
                "power_mapping": {
                    "kind": "requested_cpu_linear",
                    "kw_per_normalized_cpu": float("inf"),
                    "utilization_fraction": 1.0,
                }
            },
            "power mapping values must be finite",
        ),
        (
            {
                "deadline_mapping": {
                    "kind": "observed_runtime_multiplier",
                    "multiplier": float("inf"),
                }
            },
            "deadline multiplier must be finite",
        ),
    )
    for updates, message in invalid_cases:
        _write_config(config, source, **updates)
        with pytest.raises(ConfigurationError, match=message):
            GoogleV3ConversionConfig.from_yaml(config)


def test_google_v3_verifier_rejects_artifact_and_semantic_tampering(tmp_path: Path) -> None:
    source, config, query = _inputs(tmp_path)
    output = tmp_path / "output"
    convert_google_v3_export(source, config, query, output)

    (output / "workload.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Checksum mismatch"):
        verify_google_v3_conversion(output)

    shutil_target = tmp_path / "semantic"
    convert_google_v3_export(source, config, query, shutil_target)
    manifest_path = shutil_target / "conversion-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["rows"] = 3
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_checksums(shutil_target)
    with pytest.raises(ConfigurationError, match="row count"):
        verify_google_v3_conversion(shutil_target)

    extra = tmp_path / "extra"
    convert_google_v3_export(source, config, query, extra)
    (extra / "undeclared.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="extra"):
        verify_google_v3_conversion(extra)

    source_check = tmp_path / "source-check"
    convert_google_v3_export(source, config, query, source_check)
    changed_source = tmp_path / "changed-source.csv"
    changed_source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ConfigurationError, match="source CSV hash"):
        verify_google_v3_conversion(source_check, source_csv=changed_source)


def test_google_v3_verifier_rejects_missing_directory_and_invalid_manifest(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        verify_google_v3_conversion(tmp_path / "missing")

    source, config, query = _inputs(tmp_path)
    output = tmp_path / "output"
    convert_google_v3_export(source, config, query, output)
    (output / "conversion-manifest.json").write_text("not json\n", encoding="utf-8")
    write_checksums(output)
    with pytest.raises(ConfigurationError, match="Invalid Google v3 conversion manifest"):
        verify_google_v3_conversion(output)


def test_google_v3_cli_converts_and_verifies_source(tmp_path: Path) -> None:
    source, config, query = _inputs(tmp_path)
    output = tmp_path / "output"
    runner = CliRunner()

    converted = runner.invoke(
        app,
        [
            "trace",
            "convert-google-v3",
            str(source),
            str(config),
            str(output),
            "--query-sql",
            str(query),
        ],
    )
    assert converted.exit_code == 0, converted.output
    assert Path(converted.stdout.strip()) == output.resolve()

    verified = runner.invoke(
        app,
        ["trace", "verify-google-v3", str(output), "--source-csv", str(source)],
    )
    assert verified.exit_code == 0, verified.output
    assert "rows=2; source=verified" in verified.stdout

    offline = runner.invoke(app, ["trace", "verify-google-v3", str(output)])
    assert offline.exit_code == 0, offline.output
    assert "source=not supplied" in offline.stdout


def test_google_v3_reference_assets_match_the_converter_contract() -> None:
    assets = _REPOSITORY_ROOT / "benchmarks" / "google_clusterdata_2019"
    query = (assets / "export_workload.sql").read_text(encoding="utf-8")
    config = GoogleV3ConversionConfig.from_yaml(assets / "conversion.example.yaml")

    assert config.dataset == "google-clusterdata-2019-v3"
    assert config.input_sha256 == "0" * 64
    assert "`google.com:google-cluster-data`.clusterdata_2019_a.instance_events" in query
    assert {"@start_time_us", "@end_time_us", "@finish_cutoff_time_us"} <= set(
        re.findall(r"@[a-z_]+", query)
    )
    for column in GOOGLE_V3_EXPORT_COLUMNS:
        assert re.search(rf"\b{re.escape(column)}\b", query)
