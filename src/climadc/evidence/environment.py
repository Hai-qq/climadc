from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

from climadc import __version__
from climadc.evidence.manifest import EnvironmentRecord

_PACKAGES = {
    "climadc": "climadc",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "pyarrow": "pyarrow",
    "pydantic": "pydantic",
    "pint": "pint",
}


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _highs_version() -> str:
    try:
        highs_core = importlib.import_module("scipy.optimize._highspy._core")

        return (
            f"{highs_core.HIGHS_VERSION_MAJOR}."
            f"{highs_core.HIGHS_VERSION_MINOR}."
            f"{highs_core.HIGHS_VERSION_PATCH}"
        )
    except (ImportError, AttributeError):
        return "bundled-version-unavailable"


def _constraints() -> tuple[str, str]:
    requirements = sorted(importlib.metadata.requires("climadc") or [])
    payload = json.dumps(requirements, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), "installed climadc metadata Requires-Dist"


def capture_environment() -> EnvironmentRecord:
    constraints_hash, constraints_source = _constraints()
    packages = {name: _version(distribution) for name, distribution in _PACKAGES.items()}
    packages["climadc"] = __version__
    return EnvironmentRecord(
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        platform=platform.platform(),
        operating_system=platform.system(),
        architecture=platform.machine() or "unknown",
        timezone=str(datetime.now().astimezone().tzinfo or "unknown"),
        packages=packages,
        highs=_highs_version(),
        random_seeds={
            "python_random": None,
            "numpy_legacy_global": None,
            "scipy_highs": "fixed-aggregate allocation tie-break; no seed option configured",
        },
        dependency_constraints_sha256=constraints_hash,
        dependency_constraints_source=constraints_source,
    )


def _git_root() -> Path | None:
    candidates = (Path.cwd(), Path(__file__).resolve().parents[3])
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def git_state() -> tuple[str, bool | None]:
    """Return the checkout commit/dirty flag, or an explicit unknown state for wheels."""

    root = _git_root()
    if root is None:
        return "unknown", None
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown", None
    return commit if commit else "unknown", bool(status.strip())


__all__ = ["capture_environment", "git_state"]
