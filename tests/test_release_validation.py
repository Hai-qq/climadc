from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from pathlib import PurePosixPath
import subprocess
import sys
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


def _distributions(
    directory: Path,
    *,
    wheel_version: str = VERSION,
    forbidden_sdist_path: str | None = None,
) -> None:
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
        if forbidden_sdist_path is not None:
            _write_member(archive, f"{root}/{forbidden_sdist_path}")


@contextmanager
def _runtime_packaging_probes(repository: Path, marker_name: str) -> Iterator[None]:
    created_files: list[Path] = []
    created_directories: list[Path] = []
    marker_directories = [
        repository / "runs" / marker_name,
        repository / ".cache" / marker_name,
        repository / "data" / "raw" / marker_name,
        repository / "artifacts" / marker_name,
        repository / ".worktrees" / marker_name,
        repository / ".superpowers" / marker_name,
        repository / "site" / marker_name,
        repository / "dist" / marker_name,
        repository / "htmlcov" / marker_name,
    ]

    def create_directory(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=False)
        created_directories.append(path)

    def create_file(path: Path, payload: bytes = b"must not ship\n") -> None:
        stream = path.open("xb")
        created_files.append(path)
        with stream:
            stream.write(payload)

    try:
        for directory in marker_directories:
            create_directory(directory)
            create_file(directory / "private-runtime.txt")
        residue_directory = repository / f"{marker_name}.egg-info"
        create_directory(residue_directory)
        create_file(residue_directory / "PKG-INFO")
        for suffix in (".pyc", ".pyo", ".pyd"):
            create_file(repository / f"{marker_name}{suffix}")
        yield
    finally:
        for path in reversed(created_files):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        for path in reversed(created_directories):
            try:
                path.rmdir()
            except FileNotFoundError:
                pass


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


def test_release_validation_rejects_runtime_paths_in_sdist(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _distributions(dist, forbidden_sdist_path="runs/private-run/metrics.json")

    with pytest.raises(ReleaseValidationError, match="forbidden runtime paths"):
        validate_release(TAG, Path("pyproject.toml"), dist, None)


@pytest.mark.parametrize(
    "forbidden_path",
    [
        "climadc.egg-info/PKG-INFO",
        "nested/pkg.egg-info/file",
        "module.pyc",
        "module.pyo",
        "module.pyd",
        "__pycache__/module.pyc",
    ],
)
def test_release_validation_rejects_python_build_residue(
    tmp_path: Path,
    forbidden_path: str,
) -> None:
    dist = tmp_path / "dist"
    _distributions(dist, forbidden_sdist_path=forbidden_path)

    with pytest.raises(ReleaseValidationError, match="forbidden runtime paths"):
        validate_release(TAG, Path("pyproject.toml"), dist, None)


def test_sdist_hatch_policy_declares_runtime_exclusions() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.hatch.build.targets.sdist]" in pyproject
    for exclusion in (
        "/runs/**",
        "/.cache/**",
        "/data/raw/**",
        "/artifacts/**",
        "/.worktrees/**",
        "/.superpowers/**",
        "/site/**",
        "/dist/**",
        "/.coverage",
        "/htmlcov/**",
        "/docs/superpowers/**",
        "/**/*.egg-info/**",
        "/**/*.py[cod]",
    ):
        assert f'"{exclusion}"' in pyproject


def test_packaging_probe_collision_preserves_preexisting_marker(tmp_path: Path) -> None:
    marker_name = "collision"
    preexisting = tmp_path / ".cache" / marker_name
    preexisting.mkdir(parents=True)
    payload = preexisting / "private-runtime.txt"
    payload.write_bytes(b"owner data\n")

    with pytest.raises(FileExistsError):
        with _runtime_packaging_probes(tmp_path, marker_name):
            pytest.fail("probe setup must stop at the collision")

    assert payload.read_bytes() == b"owner data\n"
    assert preexisting.is_dir()
    assert not (tmp_path / "runs" / marker_name).exists()


def test_fresh_sdist_excludes_ignored_runtime_files_but_retains_public_sources(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    marker_name = f".climadc-packaging-test-{tmp_path.name}"

    with _runtime_packaging_probes(repository, marker_name):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(tmp_path),
            ],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        sdist = next(tmp_path.glob("*.tar.gz"))
        root = sdist.name.removesuffix(".tar.gz")
        with tarfile.open(sdist, mode="r:gz") as archive:
            relatives = [
                name.removeprefix(f"{root}/")
                for name in archive.getnames()
                if name.startswith(f"{root}/")
            ]

        forbidden_prefixes = (
            "runs/",
            ".cache/",
            "data/raw/",
            "artifacts/",
            ".worktrees/",
            ".superpowers/",
            "site/",
            "dist/",
            "htmlcov/",
            "docs/superpowers/",
        )
        forbidden = sorted(
            relative
            for relative in relatives
            if any(relative.startswith(prefix) for prefix in forbidden_prefixes)
            or relative == ".coverage"
            or any(part.endswith(".egg-info") for part in PurePosixPath(relative).parts)
            or PurePosixPath(relative).suffix in {".pyc", ".pyo", ".pyd"}
        )
        assert forbidden == []
        for required in (
            "LICENSE",
            "NOTICE",
            "docs/index.md",
            "examples/weatherdc_kasetsart/README.md",
            "src/climadc/reference/fixtures/gb_london_24h/study.yaml",
            "src/climadc/reference/fixtures/gb_london_24h/suite.yaml",
            "src/climadc/reference/fixtures/gb_london_24h/study-cost-dominant.yaml",
            "src/climadc/reference/fixtures/gb_london_24h/study-carbon-dominant.yaml",
            "src/climadc/reference/fixtures/gb_london_24h/study-demand-charge.yaml",
            "src/climadc/reference/fixtures/gb_london_24h/source-manifest.yaml",
            "src/climadc/reference/fixtures/gb_london_24h/grid-signals.csv",
            "tests/test_package.py",
            "tests/fixtures/weatherdc_small/PROVENANCE.yaml",
        ):
            assert required in relatives

        wheel = next(tmp_path.glob("*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            wheel_members = set(archive.namelist())
        for required in (
            "climadc/reference/fixtures/gb_london_24h/study.yaml",
            "climadc/reference/fixtures/gb_london_24h/suite.yaml",
            "climadc/reference/fixtures/gb_london_24h/study-cost-dominant.yaml",
            "climadc/reference/fixtures/gb_london_24h/study-carbon-dominant.yaml",
            "climadc/reference/fixtures/gb_london_24h/study-demand-charge.yaml",
            "climadc/reference/fixtures/gb_london_24h/source-manifest.yaml",
            "climadc/reference/fixtures/gb_london_24h/climate-forecast.csv",
            "climadc/reference/fixtures/gb_london_24h/actual-weather.csv",
            "climadc/reference/fixtures/gb_london_24h/grid-signals.csv",
            "climadc/reference/fixtures/gb_london_24h/workload.csv",
        ):
            assert required in wheel_members


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
