from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import uuid
from datetime import timezone
from html.parser import HTMLParser
from math import isfinite
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from climadc import __version__
from climadc.benchmark import RunResult
from climadc.contracts.frames import PREDICTION_COLUMNS
from climadc.errors import ConfigurationError
from climadc.reporting.html import render_report

REQUIRED_ARTIFACTS = frozenset(
    {
        "run.yaml",
        "lineage.json",
        "splits.parquet",
        "predictions.parquet",
        "metrics.json",
        "leakage-report.json",
        "dataset-card.md",
        "report.html",
    }
)
_SPLIT_COLUMNS = ("split_id", "partition", "position", "timestamp")


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return cast(object, cast(Any, value).item())
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_text(payload: object) -> str:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    )


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


def _load_json(path: Path) -> object:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )


def _require_mapping(payload: object, name: str) -> dict[str, object]:
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{name} must contain one object with string keys")
    return cast(dict[str, object], payload)


def _require_finite(value: object) -> None:
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("artifact contains a non-finite number")
    if isinstance(value, dict):
        for key, nested in value.items():
            _require_finite(key)
            _require_finite(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _require_finite(nested)


def _require_finite_column(
    frame: pd.DataFrame,
    column: str,
    artifact: str,
    *,
    nullable: bool,
) -> None:
    series = frame[column]
    if not pd.api.types.is_numeric_dtype(series.dtype):
        raise ValueError(f"{artifact} column {column!r} must be numeric and finite")
    if not nullable and bool(series.isna().any()):
        raise ValueError(f"{artifact} column {column!r} must be finite and non-null")
    present = series.dropna().to_numpy(dtype=float)
    if not np.isfinite(present).all():
        raise ValueError(f"{artifact} column {column!r} must contain only finite values")


def _validate_splits_frame(frame: pd.DataFrame) -> None:
    if list(frame.columns) != list(_SPLIT_COLUMNS):
        raise ValueError(f"splits.parquet must have exact columns {list(_SPLIT_COLUMNS)}")
    if frame.empty:
        raise ValueError("splits.parquet must not be empty")
    _require_finite_column(frame, "position", "splits.parquet", nullable=False)
    if not pd.api.types.is_integer_dtype(frame["position"].dtype):
        raise ValueError("splits.parquet column 'position' must have an integer dtype")
    if bool((frame["position"] < 0).any()):
        raise ValueError("splits.parquet column 'position' must be nonnegative")
    for column in ("split_id", "partition"):
        invalid = frame[column].map(lambda value: not isinstance(value, str) or not value)
        if bool(invalid.any()):
            raise ValueError(f"splits.parquet column {column!r} must contain non-empty strings")
    if not set(frame["partition"]).issubset({"train", "gap", "calibration", "test"}):
        raise ValueError("splits.parquet contains an invalid partition label")
    dtype = frame["timestamp"].dtype
    if (
        not isinstance(dtype, pd.DatetimeTZDtype)
        or str(dtype.tz) != "UTC"
        or bool(frame["timestamp"].isna().any())
    ):
        raise ValueError("splits.parquet timestamp must contain non-null exact UTC timestamps")


def _validate_predictions_frame(frame: pd.DataFrame) -> None:
    if list(frame.columns) != list(PREDICTION_COLUMNS):
        raise ValueError(f"predictions.parquet must have exact columns {list(PREDICTION_COLUMNS)}")
    if frame.empty:
        raise ValueError("predictions.parquet must not be empty")
    _require_finite_column(frame, "value", "predictions.parquet", nullable=False)
    _require_finite_column(frame, "quantile", "predictions.parquet", nullable=True)


class _ArtifactHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.add(tag)


def _atomic_pointer_path(runs_dir: Path) -> Path:
    return runs_dir / f".latest-{uuid.uuid4().hex}"


def _direct_child_name(target: str) -> str:
    windows_path = PureWindowsPath(target)
    if (
        not target
        or target in {".", ".."}
        or Path(target).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or "/" in target
        or "\\" in target
        or Path(target).name != target
    ):
        raise ConfigurationError("Run pointer target must be one direct child name")
    return target


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _is_reparse_point(status: os.stat_result) -> bool:
    attributes = int(getattr(status, "st_file_attributes", 0))
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & flag)


def _require_real_directory(path: Path) -> None:
    try:
        status = os.lstat(path)
    except OSError as exc:
        raise ConfigurationError(f"Run target must be an existing real directory: {path}") from exc
    if not stat.S_ISDIR(status.st_mode) or _is_reparse_point(status):
        raise ConfigurationError(f"Run target must be a real directory, not a link: {path}")


def _require_direct_child(runs_dir: Path, run_path: Path) -> tuple[Path, str]:
    parent = _absolute(runs_dir)
    candidate = _absolute(run_path)
    name = _direct_child_name(candidate.name)
    if candidate.parent != parent:
        raise ConfigurationError("Run path must be an existing direct child of runs directory")
    _require_real_directory(candidate)
    return candidate, name


def _resolve_direct_child(parent: Path, target: str) -> Path:
    name = _direct_child_name(target)
    candidate = _absolute(parent) / name
    _require_real_directory(candidate)
    return candidate


def update_latest_pointer(
    runs_dir: Path,
    run_path: Path,
    *,
    windows: bool | None = None,
) -> Path:
    """Atomically update latest using a relative symlink or Windows text pointer."""

    runs_dir = Path(runs_dir)
    run_path = Path(run_path)
    windows = os.name == "nt" if windows is None else windows
    latest = runs_dir / "latest"
    temporary = _atomic_pointer_path(runs_dir)
    _, relative = _require_direct_child(runs_dir, run_path)
    try:
        if windows:
            temporary.write_text(f"{relative}\n", encoding="utf-8", newline="\n")
        else:
            temporary.symlink_to(relative, target_is_directory=True)
        os.replace(temporary, latest)
    except OSError as exc:
        if temporary.is_symlink() or temporary.exists():
            temporary.unlink()
        raise ConfigurationError(f"Unable to update latest pointer {latest}: {exc}") from exc
    return latest


def resolve_run_path(path: Path, *, windows: bool | None = None) -> Path:
    """Resolve an actual run directory, POSIX symlink, or Windows text pointer."""

    pointer = _absolute(Path(path))
    windows = os.name == "nt" if windows is None else windows
    try:
        pointer_status = os.lstat(pointer)
    except OSError as exc:
        raise ConfigurationError(f"Run path does not exist: {pointer}") from exc
    if stat.S_ISDIR(pointer_status.st_mode) and not _is_reparse_point(pointer_status):
        return pointer
    if windows:
        if stat.S_ISLNK(pointer_status.st_mode) or _is_reparse_point(pointer_status):
            raise ConfigurationError("Windows text run pointer required; symlink rejected")
        if not stat.S_ISREG(pointer_status.st_mode):
            raise ConfigurationError(f"Run path does not exist: {pointer}")
        try:
            raw_target = pointer.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ConfigurationError(f"Unable to read run pointer {pointer}: {exc}") from exc
        lines = raw_target.splitlines()
        if len(lines) != 1 or lines[0] != lines[0].strip() or not lines[0]:
            raise ConfigurationError(f"Run pointer is empty: {pointer}")
        return _resolve_direct_child(pointer.parent, lines[0])
    if not stat.S_ISLNK(pointer_status.st_mode):
        if stat.S_ISREG(pointer_status.st_mode):
            raise ConfigurationError("POSIX symlink run pointer required; text pointer rejected")
        raise ConfigurationError(f"Run path does not exist: {pointer}")
    try:
        target = os.readlink(pointer)
    except OSError as exc:
        raise ConfigurationError(f"Unable to read run pointer {pointer}: {exc}") from exc
    return _resolve_direct_child(pointer.parent, target)


def _dataset_cards_markdown(result: RunResult) -> str:
    sections = ["# Dataset cards", ""]
    for card in result.dataset_cards:
        payload = yaml.safe_dump(
            card.model_dump(mode="json"),
            sort_keys=True,
            allow_unicode=False,
        ).rstrip()
        sections.extend(
            [
                f"## {card.name}",
                "",
                "```yaml",
                payload,
                "```",
                "",
            ]
        )
    return "\n".join(sections)


class ArtifactWriter:
    def _write_payloads(self, directory: Path, result: RunResult, run_id: str) -> None:
        _validate_splits_frame(result.splits)
        prediction_frame = result.predictions.to_pandas()
        _validate_predictions_frame(prediction_frame)
        started_at = result.started_at.isoformat()
        run_manifest: dict[str, Any] = {
            "run_id": run_id,
            "study_id": result.study_id,
            "climadc_version": __version__,
            "started_at": started_at,
            "config_sha256": result.config_sha256,
            "config": result.config_snapshot,
            "input_hashes": result.input_hashes,
        }
        (directory / "run.yaml").write_text(
            yaml.safe_dump(run_manifest, sort_keys=True, allow_unicode=False),
            encoding="utf-8",
            newline="\n",
        )
        lineage = {
            "run_id": run_id,
            "study_id": result.study_id,
            "climadc_version": __version__,
            "started_at": started_at,
            "config_sha256": result.config_sha256,
            "config": result.config_snapshot,
            "input_hashes": result.input_hashes,
            "split_ids": sorted(str(value) for value in result.splits["split_id"].unique()),
            "model_ids": sorted(str(value) for value in prediction_frame["model_id"].unique()),
        }
        (directory / "lineage.json").write_text(_json_text(lineage), encoding="utf-8", newline="\n")
        result.splits.to_parquet(directory / "splits.parquet", index=False)
        prediction_frame.to_parquet(directory / "predictions.parquet", index=False)
        (directory / "metrics.json").write_text(
            _json_text(result.metrics), encoding="utf-8", newline="\n"
        )
        leakage = {
            "decision_time": result.leakage_audit.decision_time.isoformat(),
            "accepted_rows": result.leakage_audit.accepted_rows,
            "rejected_rows": result.leakage_audit.rejected_rows,
            "violations": list(result.leakage_audit.violations),
        }
        (directory / "leakage-report.json").write_text(
            _json_text(leakage), encoding="utf-8", newline="\n"
        )
        (directory / "dataset-card.md").write_text(
            _dataset_cards_markdown(result), encoding="utf-8", newline="\n"
        )
        (directory / "report.html").write_text(
            render_report(result, run_id), encoding="utf-8", newline="\n"
        )

    @staticmethod
    def _validate(directory: Path, result: RunResult, run_id: str) -> None:
        actual = {item.name for item in directory.iterdir()}
        if actual != REQUIRED_ARTIFACTS:
            missing = sorted(REQUIRED_ARTIFACTS.difference(actual))
            extra = sorted(actual.difference(REQUIRED_ARTIFACTS))
            raise ConfigurationError(
                f"Invalid artifact set before publish: missing={missing}, extra={extra}"
            )
        empty = sorted(item.name for item in directory.iterdir() if item.stat().st_size == 0)
        if empty:
            raise ConfigurationError(f"Empty artifacts before publish: {empty}")

        try:
            run_manifest = yaml.safe_load((directory / "run.yaml").read_text(encoding="utf-8"))
            lineage = _load_json(directory / "lineage.json")
            metrics = _load_json(directory / "metrics.json")
            leakage = _load_json(directory / "leakage-report.json")
            splits = pd.read_parquet(directory / "splits.parquet")
            predictions = pd.read_parquet(directory / "predictions.parquet")
            _validate_splits_frame(splits)
            _validate_predictions_frame(predictions)
            cards = (directory / "dataset-card.md").read_text(encoding="utf-8")
            report = (directory / "report.html").read_text(encoding="utf-8")

            expected_provenance = {
                "run_id": run_id,
                "study_id": result.study_id,
                "climadc_version": __version__,
                "started_at": result.started_at.isoformat(),
                "config_sha256": result.config_sha256,
                "config": result.config_snapshot,
                "input_hashes": result.input_hashes,
            }
            checked_manifest = _require_mapping(run_manifest, "run.yaml")
            checked_lineage = _require_mapping(lineage, "lineage.json")
            for name, payload in (
                ("run.yaml", checked_manifest),
                ("lineage.json", checked_lineage),
            ):
                for key, expected in expected_provenance.items():
                    if payload.get(key) != expected:
                        raise ValueError(f"{name} has incorrect {key}")
            expected_lineage = {
                "split_ids": sorted(str(value) for value in result.splits["split_id"].unique()),
                "model_ids": sorted(
                    str(value) for value in result.predictions.to_pandas()["model_id"].unique()
                ),
            }
            for key, expected in expected_lineage.items():
                if checked_lineage.get(key) != expected:
                    raise ValueError(f"lineage.json has incorrect {key}")
            if metrics != result.metrics:
                raise ValueError("metrics.json does not match run result")
            expected_leakage = {
                "decision_time": result.leakage_audit.decision_time.isoformat(),
                "accepted_rows": result.leakage_audit.accepted_rows,
                "rejected_rows": result.leakage_audit.rejected_rows,
                "violations": list(result.leakage_audit.violations),
            }
            if leakage != expected_leakage:
                raise ValueError("leakage-report.json does not match run result")
            pd.testing.assert_frame_equal(
                splits.reset_index(drop=True),
                result.splits.reset_index(drop=True),
                check_dtype=True,
            )
            pd.testing.assert_frame_equal(
                predictions.loc[:, list(PREDICTION_COLUMNS)].reset_index(drop=True),
                result.predictions.to_pandas().reset_index(drop=True),
                check_dtype=True,
            )
            if cards != _dataset_cards_markdown(result):
                raise ValueError("dataset-card.md does not match frozen DatasetCard semantics")
            parser = _ArtifactHTMLParser()
            parser.feed(report)
            parser.close()
            if not {"html", "body", "h1", "pre"}.issubset(parser.tags) or "script" in parser.tags:
                raise ValueError("report.html has invalid document structure")
            if report != render_report(result, run_id):
                raise ValueError("report.html does not match frozen RunResult semantics")
            for path in directory.iterdir():
                if path.suffix in {".yaml", ".json", ".md", ".html"}:
                    text = path.read_text(encoding="utf-8")
                    if "example.invalid" in text or "placeholder" in text.lower():
                        raise ValueError(f"{path.name} contains placeholder content")
            _require_finite(run_manifest)
            _require_finite(lineage)
            _require_finite(metrics)
            _require_finite(leakage)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(f"Invalid artifact content: {exc}") from exc

    def write(self, result: RunResult, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        if result.started_at.tzinfo is None or result.started_at.utcoffset() is None:
            raise ConfigurationError("RunResult.started_at must be timezone-aware")
        timestamp = result.started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}-{result.config_sha256[:8]}"
        final = output_dir / run_id
        temporary: Path | None = None
        published = False
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            if final.exists():
                raise ConfigurationError(f"Run directory already exists: {final}")
            temporary = Path(tempfile.mkdtemp(prefix=".climadc-run-", dir=output_dir))
            self._write_payloads(temporary, result, run_id)
            self._validate(temporary, result, run_id)
            temporary.rename(final)
            published = True
            update_latest_pointer(output_dir, final)
            return _absolute(final)
        except ConfigurationError:
            if published and final.exists():
                shutil.rmtree(final)
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)
            raise
        except Exception as exc:
            if published and final.exists():
                shutil.rmtree(final)
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)
            raise ConfigurationError(f"Unable to write run artifacts: {exc}") from exc
