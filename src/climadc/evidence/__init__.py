"""Independent, byte-oriented evidence contracts and verification APIs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from climadc.evidence.manifest import (
    ARTIFACT_SCHEMA_VERSION,
    EnvironmentRecord,
    RunManifest,
    SolverRecord,
)

if TYPE_CHECKING:
    from climadc.evidence.verify import VerificationReport


def verify_run(directory: Path) -> VerificationReport:
    from climadc.evidence.verify import verify_run as implementation

    return implementation(directory)


def verify_suite(directory: Path) -> VerificationReport:
    from climadc.evidence.verify import verify_suite as implementation

    return implementation(directory)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "EnvironmentRecord",
    "RunManifest",
    "SolverRecord",
    "verify_run",
    "verify_suite",
]
