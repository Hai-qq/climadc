from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

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
from climadc.evidence.manifest import Sha256

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: NonEmptyText
    exact_text: NonEmptyText
    evidence_level: Literal["E0", "E1", "E2", "E3"]
    scenario_or_study_id: NonEmptyText
    metric: NonEmptyText
    baseline: NonEmptyText
    code_commit: NonEmptyText
    code_dirty: bool
    config_sha256: Sha256
    input_hashes: dict[str, Sha256]
    output_artifact: NonEmptyText
    output_sha256: Sha256
    reproduction_command: NonEmptyText
    limitations: list[NonEmptyText] = Field(min_length=1)
    status: Literal["verified", "illustrative", "deprecated"]


class ClaimRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    claims: list[Claim] = Field(min_length=1)

    @field_validator("claims")
    @classmethod
    def claim_ids_are_unique(cls, values: list[Claim]) -> list[Claim]:
        ids = [claim.claim_id for claim in values]
        if len(ids) != len(set(ids)):
            raise ValueError("claim_id values must be unique")
        return values

    @classmethod
    def from_yaml(cls, path: Path) -> ClaimRegistry:
        try:
            payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            return cls.model_validate(payload)
        except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
            raise ConfigurationError(f"Invalid claim registry {path}: {exc}") from exc


__all__ = ["Claim", "ClaimRegistry"]
