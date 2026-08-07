from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import pandas as pd
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from climadc.errors import ConfigurationError

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConfigurationError(f"Unable to hash source artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _direct_relative_path(value: Path) -> Path:
    posix = PurePosixPath(value.as_posix())
    if (
        value.is_absolute()
        or not posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError("artifact must be a safe relative path")
    return value


class SourceTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_time_basis: Literal["provider", "scenario_assumption", "not_applicable"]
    availability_basis: Literal["retrieval", "provider", "scenario_assumption", "not_applicable"]
    note: NonEmptyText


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: NonEmptyText
    artifact: Path
    selection: NonEmptyText
    role: NonEmptyText
    provider: NonEmptyText
    request_url: NonEmptyText
    retrieved_at: datetime
    sha256: Sha256
    bytes: int = Field(gt=0)
    license: NonEmptyText
    attribution: NonEmptyText
    provenance: Literal["external_snapshot", "external_derived", "project_generated"]
    transformations: list[NonEmptyText] = Field(default_factory=list)
    timing: SourceTiming
    limitations: list[NonEmptyText] = Field(default_factory=list)

    _artifact_is_relative = field_validator("artifact")(_direct_relative_path)

    @field_validator("retrieved_at")
    @classmethod
    def retrieved_at_is_utc(cls, value: datetime) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
            or value.utcoffset() != pd.Timedelta(0)
        ):
            raise ValueError("retrieved_at must be timezone-aware UTC")
        return value


class SourceManifest(BaseModel):
    """Machine-readable provenance whose hashes bind records to local fixture files."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    study_id: NonEmptyText
    records: list[SourceRecord] = Field(min_length=1)

    @field_validator("records")
    @classmethod
    def source_ids_are_unique(cls, records: list[SourceRecord]) -> list[SourceRecord]:
        source_ids = [record.source_id for record in records]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")
        return records

    @classmethod
    def from_yaml(cls, path: Path) -> SourceManifest:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            return cls.model_validate(raw)
        except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
            raise ConfigurationError(f"Invalid source manifest {path}: {exc}") from exc

    def validate_files(self, base_dir: Path, required: set[Path]) -> dict[str, str]:
        base = base_dir.resolve()
        required_paths = {path.resolve() for path in required}
        covered: set[Path] = set()
        hashes: dict[str, str] = {}
        for record in self.records:
            path = (base / record.artifact).resolve()
            try:
                path.relative_to(base)
            except ValueError as exc:
                raise ConfigurationError(
                    f"Source artifact escapes manifest directory: {record.artifact}"
                ) from exc
            if not path.is_file():
                raise ConfigurationError(f"Source artifact does not exist: {path}")
            actual_size = path.stat().st_size
            actual_hash = sha256_file(path)
            if actual_size != record.bytes or actual_hash != record.sha256:
                raise ConfigurationError(
                    f"Source artifact integrity check failed for {record.artifact}"
                )
            covered.add(path)
            hashes[str(record.artifact)] = actual_hash

        missing = sorted(str(path) for path in required_paths.difference(covered))
        if missing:
            raise ConfigurationError(f"Source manifest does not cover replay inputs: {missing}")
        extra = sorted(str(path) for path in covered.difference(required_paths))
        if extra:
            raise ConfigurationError(f"Source manifest contains non-input artifacts: {extra}")
        return hashes
