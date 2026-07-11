from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import climadc.benchmark as benchmark_module
from climadc.benchmark import BenchmarkRunner
from climadc.cli.scaffold import scaffold_study
from climadc.config import BacktestConfig, DecisionConfig, ModelConfig, StudyConfig
from climadc.contracts.metadata import DatasetCard
from climadc.decision import DecisionResult
from climadc.errors import ConfigurationError
from climadc.validation.leakage import LeakageAudit


def _config(tmp_path: Path) -> StudyConfig:
    return StudyConfig.from_yaml(scaffold_study(tmp_path / "study"))


def _refresh_card_hash(config, input_name: str) -> None:
    input_config = getattr(config, input_name)
    payload = input_config.card.read_text(encoding="utf-8")
    old_hash = next(
        line.split(": ", 1)[1] for line in payload.splitlines() if line.startswith("sha256:")
    )
    new_hash = hashlib.sha256(input_config.path.read_bytes()).hexdigest()
    input_config.card.write_text(payload.replace(old_hash, new_hash), encoding="utf-8")


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
        ("split-000", "train"): 73,
        ("split-000", "gap"): 3,
        ("split-000", "calibration"): 12,
        ("split-000", "test"): 8,
    }
    boundaries = splits.groupby("partition")["position"].agg(["min", "max"])
    assert boundaries.loc["train"].tolist() == [0, 72]
    assert boundaries.loc["gap"].tolist() == [73, 75]
    assert boundaries.loc["calibration"].tolist() == [76, 87]
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


def test_delayed_training_rows_are_excluded_from_fit_and_split_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    telemetry = pd.read_csv(config.telemetry.path)
    telemetry.loc[0, "available_at"] = telemetry.loc[80, "event_time"]
    telemetry.to_csv(config.telemetry.path, index=False)
    _refresh_card_hash(config, "telemetry")
    histories: list[pd.DataFrame] = []
    original_history = BenchmarkRunner._history

    def capture_history(rows: pd.DataFrame) -> pd.DataFrame:
        histories.append(rows.copy(deep=True))
        return original_history(rows)

    monkeypatch.setattr(BenchmarkRunner, "_history", staticmethod(capture_history))
    result = BenchmarkRunner().run(config)

    assert len(histories) == 1
    earliest_calibration_origin = pd.Timestamp("2026-01-04 00:00:00+00:00")
    assert (histories[0]["event_time"] <= earliest_calibration_origin).all()
    assert (histories[0]["available_at"] <= earliest_calibration_origin).all()
    delayed_position = result.splits.loc[result.splits["position"] == 0, "partition"]
    assert delayed_position.tolist() == ["gap"]
    assert len(histories[0]) == 72


def test_split_train_labels_only_positions_used_by_configured_targets(tmp_path: Path) -> None:
    config = _config(tmp_path)
    telemetry = pd.read_csv(config.telemetry.path)
    telemetry.loc[0, "available_at"] = telemetry.loc[80, "event_time"]
    unused = telemetry.iloc[[1]].copy(deep=True)
    unused.loc[:, "device_id"] = "unused-meter"
    unused.loc[:, "event_time"] = telemetry.loc[0, "event_time"]
    unused.loc[:, "available_at"] = telemetry.loc[0, "event_time"]
    unused.loc[:, "metric"] = "unused_metric"
    telemetry = pd.concat([telemetry, unused], ignore_index=True)
    telemetry.to_csv(config.telemetry.path, index=False)
    _refresh_card_hash(config, "telemetry")

    result = BenchmarkRunner().run(config)

    position_zero = result.splits.loc[result.splits["position"] == 0, "partition"]
    assert position_zero.tolist() == ["gap"]


def test_test_predictions_are_issued_only_after_safe_calibration_labels(tmp_path: Path) -> None:
    result = BenchmarkRunner().run(_config(tmp_path))
    predictions = result.predictions.to_pandas()

    assert set(predictions["issue_time"]) == {pd.Timestamp("2026-01-04 15:00:00+00:00")}
    assert predictions["valid_time"].min() == pd.Timestamp("2026-01-04 16:00:00+00:00")


def test_one_calibration_origin_is_maximum_across_configured_targets(tmp_path: Path) -> None:
    config = _config(tmp_path)
    telemetry = pd.read_csv(config.telemetry.path)
    alternate = telemetry.copy(deep=True)
    alternate["device_id"] = "alternate-meter"
    alternate["metric"] = "alternate_power"
    alternate["available_at"] = (
        pd.to_datetime(alternate["event_time"], utc=True) + pd.Timedelta("30min")
    ).map(lambda value: value.isoformat())
    pd.concat([telemetry, alternate], ignore_index=True).to_csv(config.telemetry.path, index=False)
    _refresh_card_hash(config, "telemetry")
    config = config.model_copy(
        update={
            "models": [
                ModelConfig(
                    kind="persistence",
                    model_id="total",
                    params={"target": "total_power"},
                ),
                ModelConfig(
                    kind="persistence",
                    model_id="alternate",
                    params={"target": "alternate_power"},
                ),
            ]
        }
    )

    result = BenchmarkRunner().run(config)

    expected_origin = pd.Timestamp("2026-01-04 15:30:00+00:00")
    assert set(result.predictions.to_pandas()["issue_time"]) == {expected_origin}
    assert result.metrics["decision"]["origin"] == expected_origin.isoformat()


def test_calibration_rejects_label_available_at_first_test_target(tmp_path: Path) -> None:
    config = _config(tmp_path)
    telemetry = pd.read_csv(config.telemetry.path)
    telemetry.loc[87, "available_at"] = telemetry.loc[88, "event_time"]
    telemetry.to_csv(config.telemetry.path, index=False)
    _refresh_card_hash(config, "telemetry")

    with pytest.raises(ConfigurationError, match="calibration.*before first test target"):
        BenchmarkRunner().run(config)


def test_calibration_requires_exact_observed_targets(tmp_path: Path) -> None:
    config = _config(tmp_path)
    telemetry = pd.read_csv(config.telemetry.path)
    telemetry.loc[80, "quality"] = "imputed"
    telemetry.to_csv(config.telemetry.path, index=False)
    _refresh_card_hash(config, "telemetry")

    with pytest.raises(ConfigurationError, match="exactly one observed target"):
        BenchmarkRunner().run(config)


def test_decision_comparison_uses_one_common_origin_and_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    issue_times: list[set[pd.Timestamp]] = []
    quantiles: list[list[object]] = []
    workloads: list[pd.DataFrame] = []
    forecast_values: list[pd.DataFrame] = []
    original_solve = benchmark_module.ShadowScheduler.solve

    def capture_solve(self, forecast, workload, constraints):
        frame = forecast.to_pandas()
        issue_times.append(set(frame["issue_time"]))
        quantiles.append(frame["quantile"].tolist())
        workloads.append(workload.to_pandas())
        forecast_values.append(frame)
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
    assert decision_metrics["origin"] == "2026-01-04T15:00:00+00:00"
    assert decision_metrics["model_id"] == "persistence--split-000"
    assert issue_times == [{pd.Timestamp("2026-01-04 15:00Z")}] * 3
    assert set(quantiles[0]) == {0.5}
    assert set(quantiles[1]) == {0.9}
    assert all(pd.isna(value) for value in quantiles[2])
    assert all(frame.equals(workloads[0]) for frame in workloads[1:])
    workload = workloads[0]
    assert not workload.empty
    assert (workload["available_at"] <= pd.Timestamp("2026-01-04 15:00Z")).all()
    assert (workload["deadline"] >= pd.Timestamp("2026-01-04 16:00Z")).all()
    assert (workload["deadline"] <= pd.Timestamp("2026-01-04 23:00Z")).all()
    telemetry = pd.read_csv(_config(tmp_path / "oracle").telemetry.path)
    telemetry["event_time"] = pd.to_datetime(telemetry["event_time"], utc=True)
    expected = telemetry.loc[
        telemetry["event_time"].between(
            pd.Timestamp("2026-01-04 16:00Z"), pd.Timestamp("2026-01-04 23:00Z")
        ),
        "value",
    ].to_numpy()
    assert forecast_values[2]["value"].to_numpy() == pytest.approx(expected)


def test_decision_rejects_workload_deadline_truncated_by_test_window(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workload = pd.read_csv(config.workload.path)
    workload.loc[87, "deadline"] = "2026-01-05T00:00:00+00:00"
    workload.to_csv(config.workload.path, index=False)
    _refresh_card_hash(config, "workload")

    with pytest.raises(ConfigurationError, match="deadline.*test window"):
        BenchmarkRunner().run(config)


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
    ]
    assert result.predictions.to_pandas()["model_id"].nunique() == 8
    assert result.decision is None
    assert "decision" not in result.metrics
    for split_id, rows in result.splits.groupby("split_id"):
        train = rows.loc[rows["partition"] == "train", "position"]
        calibration = rows.loc[rows["partition"] == "calibration", "position"]
        assert len(train) >= 64, split_id
        earliest_origin_position = int(calibration.min()) - 4
        assert int(train.max()) <= earliest_origin_position


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


def test_run_captures_entry_clock_and_frozen_config(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    original_load = BenchmarkRunner._load_and_validate
    entered_load_at: list[datetime] = []

    def mutate_original_after_entry(self, frozen_config):
        entered_load_at.append(datetime.now(timezone.utc))
        config.study_id = "mutated-after-entry"
        assert frozen_config is not config
        return original_load(self, frozen_config)

    monkeypatch.setattr(BenchmarkRunner, "_load_and_validate", mutate_original_after_entry)
    result = BenchmarkRunner().run(config)

    assert result.started_at <= entered_load_at[0]
    assert result.study_id == "climadc-synthetic-demo"
    assert result.config_snapshot["study_id"] == "climadc-synthetic-demo"


def test_run_rejects_input_mutated_during_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    original_read = benchmark_module.read_workload

    def mutate_after_read(*args, **kwargs):
        loaded = original_read(*args, **kwargs)
        config.workload.path.write_bytes(config.workload.path.read_bytes() + b"\n")
        return loaded

    monkeypatch.setattr(benchmark_module, "read_workload", mutate_after_read)
    with pytest.raises(ConfigurationError, match="changed during benchmark entry"):
        BenchmarkRunner().run(config)


def test_run_result_reconstructs_all_nested_inputs(tmp_path: Path) -> None:
    original = BenchmarkRunner().run(_config(tmp_path))
    splits = original.splits.copy(deep=True)
    splits["metadata"] = [{"value": 1} for _ in range(len(splits))]
    predictions = original.predictions
    metrics = {"nested": {"value": 1}}
    violation = {"index": "0", "available_at": "2026-01-01T00:00:00+00:00"}
    leakage = LeakageAudit(pd.Timestamp("2026-01-01T00:00:00Z"), 1, 1, (violation,))
    cards: tuple[DatasetCard, ...] = original.dataset_cards
    decision_schedule = original.decision.schedule.copy(deep=True)
    decision_schedule["metadata"] = [{"value": 1} for _ in range(len(decision_schedule))]
    decision_metrics = dict(original.decision.metrics)
    decision = DecisionResult(decision_schedule, True, (), decision_metrics)
    result = replace(
        original,
        splits=splits,
        predictions=predictions,
        metrics=metrics,
        leakage_audit=leakage,
        dataset_cards=cards,
        decision=decision,
    )

    splits.loc[0, "partition"] = "mutated"
    splits.loc[0, "metadata"]["value"] = 2
    predictions.to_pandas(copy=False).loc[0, "value"] = -999.0
    metrics["nested"]["value"] = 2
    violation["index"] = "mutated"
    cards[0].name = "mutated"
    decision.schedule.loc[0, "forecast_value"] = -999.0
    decision.schedule.loc[0, "metadata"]["value"] = 2
    decision.metrics["baseline_peak"] = -999.0

    assert result.splits.loc[0, "partition"] != "mutated"
    assert result.splits.loc[0, "metadata"] == {"value": 1}
    assert (result.predictions.to_pandas()["value"] >= 0).all()
    assert result.metrics["nested"] == {"value": 1}
    assert result.leakage_audit.violations[0]["index"] == "0"
    assert result.dataset_cards[0].name != "mutated"
    assert (result.decision.schedule["forecast_value"] >= 0).all()
    assert result.decision.schedule.loc[0, "metadata"] == {"value": 1}
    assert result.decision.metrics["baseline_peak"] != -999.0
