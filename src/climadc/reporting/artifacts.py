from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from climadc import __version__
from climadc.benchmark import RunResult
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


def _atomic_pointer_path(runs_dir: Path) -> Path:
    return runs_dir / f".latest-{uuid.uuid4().hex}"


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
    relative = os.path.relpath(run_path, runs_dir)
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
    if pointer.is_symlink() or pointer.is_dir():
        resolved = pointer.resolve()
    elif pointer.is_file():
        try:
            target = pointer.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise ConfigurationError(f"Unable to read run pointer {pointer}: {exc}") from exc
        if not target:
            raise ConfigurationError(f"Run pointer is empty: {pointer}")
        resolved = (pointer.parent / target).resolve()
    else:
        raise ConfigurationError(f"Run path does not exist: {pointer}")
    if not resolved.is_dir():
        raise ConfigurationError(f"Resolved run directory does not exist: {resolved}")
    return resolved


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
    def _validate(directory: Path) -> None:
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
            self._validate(temporary)
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
