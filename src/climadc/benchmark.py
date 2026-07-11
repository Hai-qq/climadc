from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from climadc.adapters import read_climate, read_telemetry, read_workload
from climadc.backtesting import TemporalSplit, rolling_origin_splits
from climadc.calibration import SplitConformalCalibrator
from climadc.config import InputConfig, ModelConfig, StudyConfig
from climadc.contracts.frames import (
    PREDICTION_COLUMNS,
    ClimateForecastFrame,
    DCTelemetryFrame,
    PredictionFrame,
    WorkloadFrame,
)
from climadc.contracts.metadata import DatasetCard
from climadc.decision import DecisionConstraints, DecisionResult, ShadowScheduler
from climadc.errors import ConfigurationError
from climadc.evaluation import point_metrics, probabilistic_metrics
from climadc.forecasting import (
    ClimatologyForecaster,
    Forecaster,
    LinearForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from climadc.forecasting.lightgbm import LightGBMForecaster
from climadc.validation.leakage import LeakageAudit, LeakageGuard

_ALPHA = 0.2


@dataclass(frozen=True)
class LoadedStudyData:
    climate: ClimateForecastFrame
    telemetry: DCTelemetryFrame
    workload: WorkloadFrame | None
    cards: tuple[DatasetCard, ...]


@dataclass(frozen=True)
class RunResult:
    study_id: str
    config_sha256: str
    config_snapshot: dict[str, object]
    input_hashes: dict[str, str]
    started_at: datetime
    splits: pd.DataFrame
    predictions: PredictionFrame
    metrics: dict[str, object]
    leakage_audit: LeakageAudit
    decision: DecisionResult | None
    dataset_cards: tuple[DatasetCard, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_snapshot", deepcopy(self.config_snapshot))
        object.__setattr__(self, "input_hashes", dict(self.input_hashes))
        object.__setattr__(self, "splits", self.splits.copy(deep=True))
        object.__setattr__(self, "metrics", deepcopy(self.metrics))
        object.__setattr__(self, "dataset_cards", tuple(self.dataset_cards))


@dataclass(frozen=True)
class _SplitExecution:
    split_id: str
    split: TemporalSplit
    models: dict[str, Forecaster]
    calibrators: dict[str, SplitConformalCalibrator]


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ConfigurationError(f"Unable to hash input {path}: {exc}") from exc


def _normalized_config(config: StudyConfig) -> dict[str, object]:
    return cast(dict[str, object], config.model_dump(mode="json"))


def _config_sha256(snapshot: dict[str, object]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _combine_prediction_frames(frames: list[PredictionFrame]) -> PredictionFrame:
    if not frames:
        raise ConfigurationError("Benchmark produced no predictions")
    combined = pd.concat([frame.to_pandas() for frame in frames], ignore_index=True)
    selected = cast(pd.DataFrame, combined.loc[:, list(PREDICTION_COLUMNS)])
    return PredictionFrame.from_pandas(selected)


class BenchmarkRunner:
    def validate(self, config: StudyConfig) -> LoadedStudyData:
        return self._load_and_validate(config)

    def _load_and_validate(self, config: StudyConfig) -> LoadedStudyData:
        inputs: list[tuple[str, InputConfig]] = [
            ("climate", config.climate),
            ("telemetry", config.telemetry),
        ]
        if config.workload is not None:
            inputs.append(("workload", config.workload))

        cards = tuple(DatasetCard.from_yaml(item.card) for _, item in inputs)
        for (name, item), card in zip(inputs, cards, strict=True):
            actual = _file_sha256(item.path)
            if actual != card.sha256:
                raise ConfigurationError(
                    f"SHA-256 mismatch for {name} input {item.path}: "
                    f"card={card.sha256}, actual={actual}"
                )

        climate = read_climate(
            config.climate.path,
            cast(object, config.climate.format),  # type: ignore[arg-type]
            config.climate.column_map,
            config.climate.timezone,
        )
        telemetry = read_telemetry(
            config.telemetry.path,
            cast(object, config.telemetry.format),  # type: ignore[arg-type]
            config.telemetry.column_map,
            config.telemetry.timezone,
        )
        workload = None
        if config.workload is not None:
            workload = read_workload(
                config.workload.path,
                cast(object, config.workload.format),  # type: ignore[arg-type]
                config.workload.column_map,
                config.workload.timezone,
            )
        return LoadedStudyData(climate, telemetry, workload, cards)

    def _make_splits(
        self, config: StudyConfig, data: LoadedStudyData
    ) -> tuple[pd.DatetimeIndex, list[tuple[str, TemporalSplit]], pd.DataFrame]:
        telemetry = data.telemetry.to_pandas()
        times = pd.DatetimeIndex(telemetry["event_time"].drop_duplicates().sort_values())
        backtest = config.backtest
        if backtest.strategy == "blocked":
            train_size = len(times) - backtest.calibration_size - backtest.test_size
            if train_size < backtest.min_train:
                raise ConfigurationError(
                    "Blocked split has insufficient training timestamps: "
                    f"{train_size} < {backtest.min_train}"
                )
            calibration_end = train_size + backtest.calibration_size
            temporal = TemporalSplit(
                train=np.arange(train_size, dtype=np.int64),
                calibration=np.arange(train_size, calibration_end, dtype=np.int64),
                test=np.arange(calibration_end, len(times), dtype=np.int64),
                train_end=times[train_size - 1],
                calibration_end=times[calibration_end - 1],
                test_end=times[-1],
            )
            splits = [("split-000", temporal)]
        else:
            rolling = rolling_origin_splits(
                times,
                min_train=backtest.min_train,
                calibration_size=backtest.calibration_size,
                test_size=backtest.test_size,
                step=backtest.step,
            )
            if not rolling:
                raise ConfigurationError("Rolling backtest produced no complete splits")
            splits = [(f"split-{position:03d}", split) for position, split in enumerate(rolling)]

        rows: list[dict[str, object]] = []
        for split_id, split in splits:
            for partition in ("train", "calibration", "test"):
                for position in cast(np.ndarray, getattr(split, partition)):
                    rows.append(
                        {
                            "split_id": split_id,
                            "partition": partition,
                            "position": int(position),
                            "timestamp": times[int(position)],
                        }
                    )
        split_table = pd.DataFrame.from_records(
            rows, columns=["split_id", "partition", "position", "timestamp"]
        )
        return times, splits, split_table

    @staticmethod
    def _target_for(config: ModelConfig, telemetry: pd.DataFrame) -> str:
        configured = config.params.get("target")
        if configured is not None:
            if not isinstance(configured, str) or not configured:
                raise ConfigurationError(f"Model {config.model_id!r} target must be non-empty")
            return configured
        metrics = sorted(str(value) for value in telemetry["metric"].unique())
        if len(metrics) != 1:
            raise ConfigurationError(
                f"Model {config.model_id!r} requires params.target when telemetry has "
                f"{len(metrics)} metrics"
            )
        return metrics[0]

    @staticmethod
    def _sequence_param(params: dict[str, object], name: str) -> tuple[str, ...]:
        value = params.pop(name, None)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ConfigurationError(f"Model parameter {name!r} must be a list of strings")
        return tuple(value)

    def _build_model(
        self, config: ModelConfig, telemetry: pd.DataFrame, split_id: str
    ) -> Forecaster:
        params = deepcopy(config.params)
        target = self._target_for(config, telemetry)
        params.pop("target", None)
        model_id = f"{config.model_id}--{split_id}"
        model: Forecaster
        if config.kind == "persistence":
            model = PersistenceForecaster(target=target, model_id=model_id)
        elif config.kind == "seasonal":
            period = params.pop("period", None)
            try:
                parsed_period = pd.Timedelta(cast(str, period))
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("Model parameter 'period' must be a duration") from exc
            model = SeasonalNaiveForecaster(target=target, period=parsed_period, model_id=model_id)
        elif config.kind == "climatology":
            model = ClimatologyForecaster(
                target=target,
                group_by=self._sequence_param(params, "group_by"),
                model_id=model_id,
            )
        elif config.kind == "linear":
            model = LinearForecaster(
                target=target,
                features=self._sequence_param(params, "features"),
                model_id=model_id,
            )
        else:
            model = LightGBMForecaster(
                target=target,
                features=self._sequence_param(params, "features"),
                model_id=model_id,
            )
        if params:
            raise ConfigurationError(
                f"Unsupported params for model {config.model_id!r}: {sorted(params)}"
            )
        return model

    @staticmethod
    def _history(telemetry: pd.DataFrame) -> pd.DataFrame:
        return cast(
            pd.DataFrame,
            pd.DataFrame(
                {
                    "site_id": telemetry["site_id"],
                    "valid_time": telemetry["event_time"],
                    "available_at": telemetry["available_at"],
                    "target": telemetry["metric"],
                    "value": telemetry["value"],
                    "unit": telemetry["unit"],
                }
            ),
        )

    @staticmethod
    def _actuals_for(predictions: PredictionFrame, telemetry: pd.DataFrame) -> pd.Series:
        points = predictions.to_pandas()
        if points["quantile"].notna().any():
            points = cast(pd.DataFrame, points.loc[points["quantile"] == 0.5])
        observed = cast(pd.DataFrame, telemetry.loc[telemetry["quality"] == "observed"])
        values: list[float] = []
        for row in points.itertuples(index=False):
            matches = observed.loc[
                (observed["site_id"] == row.site_id)
                & (observed["event_time"] == row.valid_time)
                & (observed["metric"] == row.target)
            ]
            if len(matches) != 1:
                raise ConfigurationError(
                    "Expected exactly one observed target for "
                    f"site={row.site_id!r}, target={row.target!r}, time={row.valid_time}"
                )
            values.append(float(matches.iloc[0]["value"]))
        return pd.Series(values, dtype=float)

    def _fit_predict_calibrate(
        self,
        config: StudyConfig,
        data: LoadedStudyData,
        times: pd.DatetimeIndex,
        splits: list[tuple[str, TemporalSplit]],
    ) -> tuple[PredictionFrame, list[_SplitExecution]]:
        telemetry = data.telemetry.to_pandas()
        horizon = pd.Timedelta(config.horizon)
        frames: list[PredictionFrame] = []
        executions: list[_SplitExecution] = []
        for split_id, split in splits:
            train_times = times[split.train]
            train_rows = telemetry.loc[telemetry["event_time"].isin(train_times)]
            history = self._history(train_rows)
            models: dict[str, Forecaster] = {}
            calibrators: dict[str, SplitConformalCalibrator] = {}
            calibration_targets = times[split.calibration]
            test_targets = times[split.test]
            for model_config in config.models:
                model = self._build_model(model_config, telemetry, split_id)
                model.fit(history, context={})
                calibration_point = model.predict(calibration_targets - horizon, horizon)
                calibration_actuals = self._actuals_for(calibration_point, telemetry)
                calibrator = SplitConformalCalibrator(alpha=_ALPHA).fit(
                    calibration_point, calibration_actuals
                )
                test_point = model.predict(test_targets - horizon, horizon)
                frames.append(calibrator.transform(test_point))
                models[model_config.model_id] = model
                calibrators[model_config.model_id] = calibrator
            executions.append(_SplitExecution(split_id, split, models, calibrators))
        return _combine_prediction_frames(frames), executions

    def _prediction_metrics(
        self,
        predictions: PredictionFrame,
        telemetry: pd.DataFrame,
    ) -> dict[str, object]:
        frame = predictions.to_pandas()
        result: dict[str, object] = {}
        for model_id, group in frame.groupby("model_id", sort=True):
            split_id = str(model_id).rsplit("--", 1)[-1]
            split_metrics = cast(dict[str, object], result.setdefault(split_id, {}))
            median = cast(pd.DataFrame, group.loc[group["quantile"] == 0.5])
            lower = cast(pd.DataFrame, group.loc[group["quantile"] == _ALPHA / 2.0])
            upper = cast(pd.DataFrame, group.loc[group["quantile"] == 1.0 - _ALPHA / 2.0])
            actual = self._actuals_for(PredictionFrame.from_pandas(median), telemetry)
            split_metrics[str(model_id)] = {
                "point": point_metrics(actual, median["value"].to_numpy()),
                "probabilistic": probabilistic_metrics(
                    actual,
                    lower["value"].to_numpy(),
                    median["value"].to_numpy(),
                    upper["value"].to_numpy(),
                    alpha=_ALPHA,
                ),
            }
        return result

    def _same_origin_forecast(
        self,
        model: Forecaster,
        calibrator: SplitConformalCalibrator,
        targets: pd.DatetimeIndex,
        origin: pd.Timestamp,
    ) -> PredictionFrame:
        frames = [model.predict(pd.DatetimeIndex([origin]), target - origin) for target in targets]
        return calibrator.transform(_combine_prediction_frames(frames))

    def _decision_comparison(
        self,
        config: StudyConfig,
        data: LoadedStudyData,
        times: pd.DatetimeIndex,
        execution: _SplitExecution,
    ) -> tuple[DecisionResult | None, dict[str, object]]:
        if not config.decision.enabled:
            return None, {}
        if data.workload is None:
            raise ConfigurationError("Decision evaluation is enabled but workload is required")

        primary_config = config.models[0]
        model = execution.models[primary_config.model_id]
        calibrator = execution.calibrators[primary_config.model_id]
        targets = times[execution.split.test]
        origin = targets[0] - pd.Timedelta(config.horizon)
        calibrated = self._same_origin_forecast(model, calibrator, targets, origin)
        frame = calibrated.to_pandas()
        point_frame = cast(pd.DataFrame, frame.loc[frame["quantile"] == 0.5])
        p90_frame = cast(pd.DataFrame, frame.loc[frame["quantile"] == 0.9])
        point = PredictionFrame.from_pandas(point_frame)
        p90 = PredictionFrame.from_pandas(p90_frame)

        actuals = self._actuals_for(point, data.telemetry.to_pandas())
        oracle_rows = point_frame.copy(deep=True)
        oracle_rows["value"] = actuals.to_numpy()
        oracle_rows["model_id"] = f"oracle--{execution.split_id}"
        oracle_rows["quantile"] = pd.NA
        oracle = PredictionFrame.from_pandas(oracle_rows)

        workload_frame = data.workload.to_pandas()
        workload_rows = workload_frame.loc[
            (workload_frame["event_time"] >= targets[0])
            & (workload_frame["event_time"] <= targets[-1])
        ]
        if workload_rows.empty:
            raise ConfigurationError("No workload rows overlap the final test window")
        workload = WorkloadFrame.from_pandas(workload_rows)
        decision_config = config.decision
        constraints = DecisionConstraints(
            flexible_fraction=decision_config.flexible_fraction,
            max_shift_multiplier=decision_config.max_shift_multiplier,
            peak_penalty=decision_config.peak_penalty,
            risk_penalty=decision_config.risk_penalty,
        )
        scheduler = ShadowScheduler()
        point_result = scheduler.solve(point, workload, constraints)
        p90_result = scheduler.solve(p90, workload, constraints)
        oracle_result = scheduler.solve(oracle, workload, constraints)

        def delta(result: DecisionResult) -> dict[str, float]:
            return {
                key: float(value - oracle_result.metrics[key])
                for key, value in result.metrics.items()
                if key in oracle_result.metrics
            }

        comparison: dict[str, object] = {
            "origin": origin.isoformat(),
            "model_id": str(point_frame["model_id"].iloc[0]),
            "point": dict(point_result.metrics),
            "p90": dict(p90_result.metrics),
            "oracle": dict(oracle_result.metrics),
            "point_vs_oracle": delta(point_result),
            "p90_vs_oracle": delta(p90_result),
        }
        return point_result, comparison

    def run(self, config: StudyConfig) -> RunResult:
        model_ids = [model.model_id for model in config.models]
        if len(model_ids) != len(set(model_ids)):
            raise ConfigurationError("Configured model_id values must be unique")
        data = self._load_and_validate(config)
        if config.decision.enabled and data.workload is None:
            raise ConfigurationError("Decision evaluation is enabled but workload is required")
        times, temporal_splits, split_table = self._make_splits(config, data)
        predictions, executions = self._fit_predict_calibrate(config, data, times, temporal_splits)
        final_split = executions[-1]
        final_targets = times[final_split.split.test]
        decision_origin = final_targets[0] - pd.Timedelta(config.horizon)
        leakage_audit = LeakageGuard().audit(data.climate.to_pandas(), decision_origin)
        metrics: dict[str, object] = {
            "predictions": self._prediction_metrics(predictions, data.telemetry.to_pandas())
        }
        decision, decision_metrics = self._decision_comparison(config, data, times, final_split)
        if config.decision.enabled:
            metrics["decision"] = decision_metrics

        snapshot = _normalized_config(config)
        input_hashes = {
            "climate": _file_sha256(config.climate.path),
            "telemetry": _file_sha256(config.telemetry.path),
        }
        if config.workload is not None:
            input_hashes["workload"] = _file_sha256(config.workload.path)
        return RunResult(
            study_id=config.study_id,
            config_sha256=_config_sha256(snapshot),
            config_snapshot=snapshot,
            input_hashes=input_hashes,
            started_at=datetime.now(timezone.utc),
            splits=split_table,
            predictions=predictions,
            metrics=metrics,
            leakage_audit=leakage_audit,
            decision=decision,
            dataset_cards=data.cards,
        )
