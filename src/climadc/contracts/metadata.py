from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError
from pydantic.functional_validators import field_validator

from climadc.errors import ConfigurationError

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _validate_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError(f"Unknown IANA timezone: {value}") from exc
    return value


class SiteMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    site_id: NonEmptyText
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    timezone: str

    _timezone_is_iana = field_validator("timezone")(_validate_timezone)


class SourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: NonEmptyText
    url: NonEmptyText
    license: NonEmptyText
    redistribution_constraints: str | None = None


class DatasetCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: NonEmptyText
    site: SiteMetadata
    source: SourceMetadata
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: NonEmptyText
    time_start: datetime | None = None
    time_end: datetime | None = None
    sampling_frequency: str | None = None
    known_missing: list[str] = Field(default_factory=list)
    spatial_mismatch: list[str] = Field(default_factory=list)
    quality_limitations: list[str] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> DatasetCard:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            return cls.model_validate(raw)
        except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
            raise ConfigurationError(f"Invalid dataset card {path}: {exc}") from exc
