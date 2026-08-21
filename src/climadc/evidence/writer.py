from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from climadc import __version__
from climadc.evidence.checksums import CHECKSUM_FILE, artifact_files, write_checksums
from climadc.evidence.environment import capture_environment, git_state
from climadc.evidence.manifest import RunManifest, SolverRecord

RUN_MANIFEST_FILE = "run-manifest.json"
ENVIRONMENT_FILE = "environment.json"


def _json_text(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False, default=str) + "\n"


def finalize_evidence(
    directory: Path,
    *,
    run_type: Literal["benchmark", "replay", "replay_suite"],
    run_id: str,
    study_id: str,
    started_at: pd.Timestamp | datetime,
    config_sha256: str,
    input_hashes: dict[str, str],
    solver: SolverRecord,
) -> RunManifest:
    """Add environment, directory manifest, then a checksum file in dependency order."""

    root = Path(directory)
    environment = capture_environment()
    (root / ENVIRONMENT_FILE).write_text(
        _json_text(environment.model_dump(mode="json")), encoding="utf-8", newline="\n"
    )
    commit, dirty = git_state()
    artifacts = sorted(
        {
            *artifact_files(root),
            RUN_MANIFEST_FILE,
            CHECKSUM_FILE,
        }
    )
    started = pd.Timestamp(started_at)
    manifest = RunManifest(
        run_type=run_type,
        run_id=run_id,
        study_id=study_id,
        climadc_version=__version__,
        git_commit=commit,
        git_dirty=dirty,
        started_at=started.to_pydatetime(),
        config_sha256=config_sha256,
        input_hashes=input_hashes,
        solver=solver,
        artifacts=artifacts,
    )
    (root / RUN_MANIFEST_FILE).write_text(
        _json_text(manifest.model_dump(mode="json")), encoding="utf-8", newline="\n"
    )
    write_checksums(root)
    return manifest


__all__ = ["ENVIRONMENT_FILE", "RUN_MANIFEST_FILE", "finalize_evidence"]
