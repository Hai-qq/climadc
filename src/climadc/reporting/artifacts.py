from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import timezone
from html import unescape
from html.parser import HTMLParser
from math import isfinite
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any, cast

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


def _require_direct_child(runs_dir: Path, run_path: Path) -> tuple[Path, str]:
    parent = runs_dir.resolve()
    candidate = run_path if run_path.is_absolute() else run_path.resolve()
    name = _direct_child_name(candidate.name)
    if run_path.parent.resolve() != parent or not candidate.is_dir():
        raise ConfigurationError("Run path must be an existing direct child of runs directory")
    resolved = candidate.resolve()
    if resolved.parent != parent:
        raise ConfigurationError("Run path must resolve to a direct child of runs directory")
    return resolved, name


def _resolve_direct_child(parent: Path, target: str) -> Path:
    name = _direct_child_name(target)
    candidate = parent / name
    if not candidate.is_dir() or candidate.resolve().parent != parent.resolve():
        raise ConfigurationError("Run pointer target must resolve to an existing direct child")
    return candidate.resolve()


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

    pointer = Path(path)
    windows = os.name == "nt" if windows is None else windows
    if pointer.is_dir() and not pointer.is_symlink():
        return pointer.resolve()
    if windows:
        if pointer.is_symlink():
            raise ConfigurationError("Windows text run pointer required; symlink rejected")
        if not pointer.is_file():
            raise ConfigurationError(f"Run path does not exist: {pointer}")
        try:
            raw_target = pointer.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ConfigurationError(f"Unable to read run pointer {pointer}: {exc}") from exc
        lines = raw_target.splitlines()
        if len(lines) != 1 or lines[0] != lines[0].strip() or not lines[0]:
            raise ConfigurationError(f"Run pointer is empty: {pointer}")
        return _resolve_direct_child(pointer.parent, lines[0])
    if not pointer.is_symlink():
        if pointer.is_file():
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
        sections.extend(
            [
                f"## {card.name}",
                "",
                f"- Site: `{card.site.site_id}`",
                f"- Provider: {card.source.provider}",
                f"- License: `{card.source.license}`",
                f"- Source: {card.source.url}",
                f"- SHA-256: `{card.sha256}`",
                f"- Schema version: `{card.schema_version}`",
                "",
            ]
        )
    return "\n".join(sections)


class ArtifactWriter:
    def _write_payloads(self, directory: Path, result: RunResult, run_id: str) -> None:
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
        prediction_frame = result.predictions.to_pandas()
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
            if not cards.startswith("# Dataset cards\n") or cards.count("\n## ") != len(
                result.dataset_cards
            ):
                raise ValueError("dataset-card.md has invalid section structure")
            for card in result.dataset_cards:
                if card.sha256 not in cards or card.source.url not in cards:
                    raise ValueError("dataset-card.md is missing exact card provenance")
            parser = _ArtifactHTMLParser()
            parser.feed(report)
            parser.close()
            if not {"html", "body", "h1", "pre"}.issubset(parser.tags) or "script" in parser.tags:
                raise ValueError("report.html has invalid document structure")
            visible_report = unescape(report)
            if run_id not in visible_report or result.study_id not in visible_report:
                raise ValueError("report.html is missing run identity")
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
            return final.resolve()
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
