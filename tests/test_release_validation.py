from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tarfile
import zipfile

import pytest
import yaml

from scripts.validate_release import (
    ReleaseValidationError,
    expected_release_tag,
    validate_release,
)


VERSION = "0.1.0a1"
TAG = "v0.1.0-alpha.1"
NAME = "climadc"


def _metadata(version: str = VERSION) -> bytes:
    return f"Metadata-Version: 2.4\nName: {NAME}\nVersion: {version}\n\n".encode()


def _write_member(archive: tarfile.TarFile, name: str, payload: bytes = b"present\n") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, BytesIO(payload))


def _distributions(directory: Path, *, wheel_version: str = VERSION) -> None:
    directory.mkdir()
    wheel = directory / f"{NAME}-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        dist_info = f"{NAME}-{VERSION}.dist-info"
        archive.writestr(f"{dist_info}/METADATA", _metadata(wheel_version))
        archive.writestr(f"{dist_info}/licenses/LICENSE", "Apache-2.0\n")
        archive.writestr(f"{dist_info}/licenses/NOTICE", "ClimaDC\n")

    root = f"{NAME}-{VERSION}"
    with tarfile.open(directory / f"{root}.tar.gz", "w:gz") as archive:
        _write_member(archive, f"{root}/PKG-INFO", _metadata())
        for relative in (
            "LICENSE",
            "NOTICE",
            "docs/index.md",
            "docs/quickstart.md",
            "tests/test_package.py",
            "tests/fixtures/weatherdc_small/PROVENANCE.yaml",
        ):
            _write_member(archive, f"{root}/{relative}")


def test_expected_release_tag_maps_pep440_alpha() -> None:
    assert expected_release_tag(VERSION) == TAG


def test_release_validation_accepts_matching_tag_metadata_and_contents(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _distributions(dist)
    github_output = tmp_path / "github-output"

    version = validate_release(TAG, Path("pyproject.toml"), dist, github_output)

    assert version == VERSION
    assert github_output.read_text(encoding="utf-8") == ("verified=true\npackage_version=0.1.0a1\n")


def test_release_validation_rejects_arbitrary_tag_before_artifact_checks(tmp_path: Path) -> None:
    output = tmp_path / "github-output"

    with pytest.raises(ReleaseValidationError, match="does not match package version"):
        validate_release("v9.9.9", Path("pyproject.toml"), None, output)

    assert not output.exists()


def test_release_validation_rejects_mismatched_wheel_metadata(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _distributions(dist, wheel_version="9.9.9")

    with pytest.raises(ReleaseValidationError, match="wheel METADATA version"):
        validate_release(TAG, Path("pyproject.toml"), dist, None)


def test_sdist_hatch_policy_excludes_only_internal_plans_from_public_docs() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.hatch.build.targets.sdist]" in pyproject
    assert 'exclude = ["/docs/superpowers/**"]' in pyproject


def test_release_workflow_passes_event_data_through_step_environment() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/release.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["build"]["steps"]
    validators = [step for step in steps if "scripts/validate_release.py" in step.get("run", "")]

    assert len(validators) == 2
    for step in validators:
        assert step["env"]["RELEASE_TAG"] == "${{ github.event.release.tag_name }}"
        assert '--tag "$RELEASE_TAG"' in step["run"]
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            assert "github.event." not in step.get("run", "")
