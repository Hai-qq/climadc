from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from climadc.evidence.checksums import safe_relative_path

ARTIFACT_SCHEMA_VERSION = "2"
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class SolverRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: NonEmptyText
    method: NonEmptyText
    options: dict[str, object] = Field(default_factory=dict)


class RunManifest(BaseModel):
    """Portable directory-level contract for one v2 run or suite."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    artifact_schema_version: Literal["2"] = "2"
    run_type: Literal["benchmark", "replay", "replay_suite"]
    run_id: NonEmptyText
    study_id: NonEmptyText
    climadc_version: NonEmptyText
    git_commit: NonEmptyText
    git_dirty: bool | None
    started_at: datetime
    config_sha256: Sha256
    input_hashes: dict[str, Sha256]
    solver: SolverRecord
    artifacts: list[NonEmptyText] = Field(min_length=1)

    @field_validator("started_at")
    @classmethod
    def started_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != pd.Timedelta(0):
            raise ValueError("started_at must be timezone-aware UTC")
        return value

    @field_validator("artifacts")
    @classmethod
    def artifacts_are_safe_unique_sorted(cls, values: list[str]) -> list[str]:
        checked = [safe_relative_path(value) for value in values]
        if len(checked) != len(set(checked)):
            raise ValueError("artifacts must be unique")
        if checked != sorted(checked):
            raise ValueError("artifacts must use stable lexical ordering")
        return checked


class EnvironmentRecord(BaseModel):
    """Runtime and dependency facts captured without secrets or host-specific paths."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    python: NonEmptyText
    implementation: NonEmptyText
    platform: NonEmptyText
    operating_system: NonEmptyText
    architecture: NonEmptyText
    timezone: NonEmptyText
    packages: dict[str, NonEmptyText]
    highs: NonEmptyText
    random_seeds: dict[str, int | str | None]
    dependency_constraints_sha256: Sha256
    dependency_constraints_source: NonEmptyText


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "EnvironmentRecord",
    "RunManifest",
    "Sha256",
    "SolverRecord",
]
