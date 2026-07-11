from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

import climadc.benchmark as benchmark_module
from climadc.benchmark import BenchmarkRunner
from climadc.cli.scaffold import scaffold_study
from climadc.config import BacktestConfig, DecisionConfig, StudyConfig
from climadc.errors import ConfigurationError


def _config(tmp_path: Path) -> StudyConfig:
    return StudyConfig.from_yaml(scaffold_study(tmp_path / "study"))


def test_scaffold_cards_trace_each_input_hash(tmp_path: Path) -> None:
    config = _config(tmp_path)
    loaded = BenchmarkRunner().validate(config)

    assert len(loaded.cards) == 3
    for input_config, card in zip(
        (config.climate, config.telemetry, config.workload), loaded.cards, strict=True
    ):
        assert input_config is not None
        assert card.sha256 == hashlib.sha256(input_config.path.read_bytes()).hexdigest()


def test_validation_rejects_dataset_card_hash_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.climate.path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="SHA-256 mismatch"):
        BenchmarkRunner().validate(config)


def test_benchmark_uses_isolated_ordered_splits_and_unique_models(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = BenchmarkRunner().run(config)
    splits = result.splits

    assert splits.groupby(["split_id", "partition"]).size().to_dict() == {
        ("split-000", "train"): 72,
        ("split-000", "calibration"): 16,
        ("split-000", "test"): 8,
    }
    boundaries = splits.groupby("partition")["position"].agg(["min", "max"])
    assert boundaries.loc["train"].tolist() == [0, 71]
    assert boundaries.loc["calibration"].tolist() == [72, 87]
    assert boundaries.loc["test"].tolist() == [88, 95]

    predictions = result.predictions.to_pandas()
    assert predictions["model_id"].nunique() == 4
    assert set(predictions["model_id"]) == {
        "persistence--split-000",
        "seasonal--split-000",
        "climatology--split-000",
        "linear--split-000",
    }
    assert not predictions.duplicated(
        ["site_id", "model_id", "issue_time", "valid_time", "target", "quantile"]
    ).any()
    assert result.leakage_audit.accepted_rows == 96
    assert result.leakage_audit.rejected_rows == 0


def test_decision_comparison_uses_one_common_origin_and_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    issue_times: list[set[pd.Timestamp]] = []
    quantiles: list[list[object]] = []
    original_solve = benchmark_module.ShadowScheduler.solve

    def capture_solve(self, forecast, workload, constraints):
        frame = forecast.to_pandas()
        issue_times.append(set(frame["issue_time"]))
        quantiles.append(frame["quantile"].tolist())
        return original_solve(self, forecast, workload, constraints)

    monkeypatch.setattr(benchmark_module.ShadowScheduler, "solve", capture_solve)
    result = BenchmarkRunner().run(_config(tmp_path))
    decision_metrics = result.metrics["decision"]

    assert result.decision is not None
    assert set(decision_metrics) == {
        "origin",
        "model_id",
        "point",
        "p90",
        "oracle",
        "point_vs_oracle",
        "p90_vs_oracle",
    }
    assert decision_metrics["origin"] == "2026-01-04T12:00:00+00:00"
    assert decision_metrics["model_id"] == "persistence--split-000"
    assert issue_times == [{pd.Timestamp("2026-01-04 12:00Z")}] * 3
    assert set(quantiles[0]) == {0.5}
    assert set(quantiles[1]) == {0.9}
    assert all(pd.isna(value) for value in quantiles[2])


def test_run_result_copies_config_and_hash_provenance(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = BenchmarkRunner().run(config)
    original = result.config_snapshot["study_id"]
    config.study_id = "mutated"

    assert result.config_snapshot["study_id"] == original
    assert set(result.input_hashes) == {"climate", "telemetry", "workload"}
    assert len(result.config_sha256) == 64


def test_decision_enabled_requires_workload(tmp_path: Path) -> None:
    config = _config(tmp_path).model_copy(update={"workload": None})
    with pytest.raises(ConfigurationError, match="workload is required"):
        BenchmarkRunner().run(config)


def test_duplicate_configured_model_ids_fail_explicitly(tmp_path: Path) -> None:
    config = _config(tmp_path)
    duplicate = config.model_copy(update={"models": [config.models[0], config.models[0]]})
    with pytest.raises(ConfigurationError, match="model_id values must be unique"):
        BenchmarkRunner().run(duplicate)


def test_rolling_backtest_uses_explicit_unique_split_ids(tmp_path: Path) -> None:
    config = _config(tmp_path).model_copy(
        update={
            "backtest": BacktestConfig(
                strategy="rolling",
                min_train=64,
                calibration_size=8,
                test_size=8,
                step=8,
            ),
            "decision": DecisionConfig(enabled=False),
        }
    )

    result = BenchmarkRunner().run(config)

    assert result.splits["split_id"].unique().tolist() == [
        "split-000",
        "split-001",
        "split-002",
    ]
    assert result.predictions.to_pandas()["model_id"].nunique() == 12
    assert result.decision is None
    assert "decision" not in result.metrics


def test_decision_disabled_allows_study_without_workload(tmp_path: Path) -> None:
    config = _config(tmp_path).model_copy(
        update={"workload": None, "decision": DecisionConfig(enabled=False)}
    )

    result = BenchmarkRunner().run(config)

    assert result.decision is None
    assert set(result.input_hashes) == {"climate", "telemetry"}


def test_leakage_audit_records_unavailable_climate_candidate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    climate = pd.read_csv(config.climate.path)
    climate.loc[95, "available_at"] = climate.loc[95, "valid_time"]
    climate.to_csv(config.climate.path, index=False)
    card = config.climate.card.read_text(encoding="utf-8")
    old_hash = next(
        line.split(": ", 1)[1] for line in card.splitlines() if line.startswith("sha256:")
    )
    new_hash = hashlib.sha256(config.climate.path.read_bytes()).hexdigest()
    config.climate.card.write_text(card.replace(old_hash, new_hash), encoding="utf-8")

    result = BenchmarkRunner().run(config)
    assert result.leakage_audit.accepted_rows == 95
    assert result.leakage_audit.rejected_rows == 1
    assert len(result.leakage_audit.violations) == 1
