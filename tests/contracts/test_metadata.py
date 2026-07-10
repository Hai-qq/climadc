from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from climadc.contracts.metadata import DatasetCard, SiteMetadata, SourceMetadata
from climadc.errors import ConfigurationError


def test_dataset_card_requires_source_license_and_timezone() -> None:
    with pytest.raises(ValidationError):
        DatasetCard(
            name="missing-metadata",
            site=SiteMetadata(
                site_id="dc-1",
                latitude=13.8,
                longitude=100.5,
                timezone="UTC",
            ),
            source=SourceMetadata(
                provider="owner",
                url="https://example.test",
                license="",
            ),
            sha256="a" * 64,
            schema_version="1.0",
        )


def test_site_metadata_rejects_invalid_timezone() -> None:
    with pytest.raises(ValidationError):
        SiteMetadata(
            site_id="dc-1",
            latitude=13.8,
            longitude=100.5,
            timezone="Mars/Olympus_Mons",
        )


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [(-90.1, 100.5), (90.1, 100.5), (13.8, -180.1), (13.8, 180.1)],
)
def test_site_metadata_rejects_out_of_bounds_coordinates(
    latitude: float,
    longitude: float,
) -> None:
    with pytest.raises(ValidationError):
        SiteMetadata(
            site_id="dc-1",
            latitude=latitude,
            longitude=longitude,
            timezone="UTC",
        )


@pytest.mark.parametrize("sha256", ["a" * 63, "A" * 64, "g" * 64])
def test_dataset_card_rejects_invalid_sha256(sha256: str) -> None:
    with pytest.raises(ValidationError):
        DatasetCard(
            name="invalid-sha",
            site=SiteMetadata(
                site_id="dc-1",
                latitude=13.8,
                longitude=100.5,
                timezone="UTC",
            ),
            source=SourceMetadata(
                provider="owner",
                url="https://example.test",
                license="CC-BY-4.0",
            ),
            sha256=sha256,
            schema_version="1.0",
        )


def test_dataset_card_accepts_optional_provenance_metadata() -> None:
    card = DatasetCard(
        name="complete-metadata",
        site=SiteMetadata(
            site_id="dc-1",
            latitude=13.8,
            longitude=100.5,
            timezone="Asia/Bangkok",
        ),
        source=SourceMetadata(
            provider="owner",
            url="https://example.test",
            license="CC-BY-4.0",
            redistribution_constraints="Retain attribution.",
        ),
        sha256="a" * 64,
        schema_version="1.0",
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-02T00:00:00Z",
        sampling_frequency="1h",
        known_missing=["2026-01-01T03:00:00Z"],
        spatial_mismatch=["Weather station is 5 km from the site."],
        quality_limitations=["Workload is synthetic."],
    )

    assert card.source.redistribution_constraints == "Retain attribution."
    assert card.sampling_frequency == "1h"


def test_dataset_card_loads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "dataset-card.yaml"
    path.write_text(
        """name: demo
site:
  site_id: dc-1
  latitude: 13.8
  longitude: 100.5
  timezone: UTC
source:
  provider: owner
  url: https://example.test
  license: CC-BY-4.0
sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
schema_version: "1.0"
""",
        encoding="utf-8",
    )

    card = DatasetCard.from_yaml(path)

    assert card.name == "demo"


def test_dataset_card_wraps_file_errors(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigurationError, match="Invalid dataset card") as exc_info:
        DatasetCard.from_yaml(path)

    assert str(path) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, OSError)


def test_dataset_card_wraps_encoding_errors(tmp_path: Path) -> None:
    path = tmp_path / "invalid-utf8.yaml"
    path.write_bytes(b"\xff")

    with pytest.raises(ConfigurationError) as exc_info:
        DatasetCard.from_yaml(path)

    assert str(path) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


def test_dataset_card_wraps_yaml_errors(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("site: [", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid dataset card") as exc_info:
        DatasetCard.from_yaml(path)

    assert str(path) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, yaml.YAMLError)


def test_dataset_card_wraps_validation_errors(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("name: incomplete", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid dataset card") as exc_info:
        DatasetCard.from_yaml(path)

    assert str(path) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValidationError)
