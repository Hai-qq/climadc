from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
import re
import tarfile
import zipfile


class ReleaseValidationError(ValueError):
    """Raised when a release tag or distribution is not publishable."""


_VERSION_PATTERN = re.compile(
    r"(?P<release>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))"
    r"(?:(?P<kind>a|b|rc)(?P<number>0|[1-9]\d*))?"
)
_PROJECT_VALUE_PATTERN = re.compile(
    r'(?P<key>name|version)\s*=\s*"(?P<value>[^"\r\n]+)"\s*(?:#.*)?'
)


def _project_identity(path: Path) -> tuple[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ReleaseValidationError(f"Unable to read {path}: {exc}") from exc
    section = ""
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip()
            continue
        if section != "project":
            continue
        match = _PROJECT_VALUE_PATTERN.fullmatch(stripped)
        if match is None:
            continue
        key = match.group("key")
        if key in values:
            raise ReleaseValidationError(f"Duplicate [project] {key} in {path}")
        values[key] = match.group("value")
    missing = sorted({"name", "version"}.difference(values))
    if missing:
        raise ReleaseValidationError(f"Missing [project] values in {path}: {', '.join(missing)}")
    return values["name"], values["version"]


def expected_release_tag(version: str) -> str:
    """Map the supported PEP 440 release form to its public Git tag."""

    match = _VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ReleaseValidationError(f"Unsupported package release version: {version}")
    tag = f"v{match.group('release')}"
    kind = match.group("kind")
    if kind is None:
        return tag
    label = {"a": "alpha", "b": "beta", "rc": "rc"}[kind]
    return f"{tag}-{label}.{match.group('number')}"


def _metadata_identity(payload: bytes, label: str) -> tuple[str, str]:
    message = BytesParser(policy=default).parsebytes(payload)
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise ReleaseValidationError(f"{label} is missing Name or Version")
    return name, version


def _require_identity(actual: tuple[str, str], expected: tuple[str, str], label: str) -> None:
    if actual[0] != expected[0]:
        raise ReleaseValidationError(
            f"{label} name {actual[0]!r} does not match package name {expected[0]!r}"
        )
    if actual[1] != expected[1]:
        raise ReleaseValidationError(
            f"{label} version {actual[1]!r} does not match package version {expected[1]!r}"
        )


def _distribution_paths(dist: Path, name: str, version: str) -> tuple[Path, Path]:
    try:
        files = sorted(path for path in dist.iterdir() if path.is_file())
    except OSError as exc:
        raise ReleaseValidationError(
            f"Unable to inspect distribution directory {dist}: {exc}"
        ) from exc
    wheels = [path for path in files if path.suffix == ".whl"]
    sdists = [path for path in files if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1 or len(files) != 2:
        raise ReleaseValidationError(
            "Distribution directory must contain exactly one wheel and one .tar.gz sdist"
        )
    normalized_name = re.sub(r"[-_.]+", "_", name).lower()
    wheel = wheels[0]
    if not wheel.name.startswith(f"{normalized_name}-{version}-"):
        raise ReleaseValidationError(
            f"Wheel filename {wheel.name!r} does not contain package version {version!r}"
        )
    sdist = sdists[0]
    expected_sdist = f"{re.sub(r'[-_.]+', '-', name).lower()}-{version}.tar.gz"
    if sdist.name != expected_sdist:
        raise ReleaseValidationError(
            f"Sdist filename {sdist.name!r} does not match {expected_sdist!r}"
        )
    return wheel, sdist


def _validate_wheel(path: Path, identity: tuple[str, str]) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ReleaseValidationError("Wheel must contain exactly one dist-info/METADATA")
            _require_identity(
                _metadata_identity(archive.read(metadata[0]), "wheel METADATA"),
                identity,
                "wheel METADATA",
            )
            for required in ("LICENSE", "NOTICE"):
                suffix = f".dist-info/licenses/{required}"
                if sum(name.endswith(suffix) for name in names) != 1:
                    raise ReleaseValidationError(f"Wheel must contain exactly one {required}")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleaseValidationError(f"Unable to inspect wheel {path}: {exc}") from exc


def _validate_sdist(path: Path, identity: tuple[str, str]) -> None:
    root = path.name.removesuffix(".tar.gz")
    required = {
        f"{root}/LICENSE",
        f"{root}/NOTICE",
        f"{root}/docs/index.md",
        f"{root}/docs/quickstart.md",
        f"{root}/tests/test_package.py",
        f"{root}/tests/fixtures/weatherdc_small/PROVENANCE.yaml",
    }
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = {member.name: member for member in archive.getmembers()}
            missing = sorted(required.difference(members))
            if missing:
                raise ReleaseValidationError(f"Sdist is missing required files: {missing}")
            internal_prefix = f"{root}/docs/superpowers/"
            if any(name.startswith(internal_prefix) for name in members):
                raise ReleaseValidationError("Sdist contains internal docs/superpowers plans")
            metadata_name = f"{root}/PKG-INFO"
            metadata_member = members.get(metadata_name)
            if metadata_member is None or not metadata_member.isfile():
                raise ReleaseValidationError("Sdist must contain one root PKG-INFO file")
            metadata_file = archive.extractfile(metadata_member)
            if metadata_file is None:
                raise ReleaseValidationError("Unable to read sdist PKG-INFO")
            _require_identity(
                _metadata_identity(metadata_file.read(), "sdist PKG-INFO"),
                identity,
                "sdist PKG-INFO",
            )
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseValidationError(f"Unable to inspect sdist {path}: {exc}") from exc


def validate_release(
    tag: str,
    pyproject: Path,
    dist: Path | None,
    github_output: Path | None,
) -> str:
    """Validate release identity and, when supplied, both distribution artifacts."""

    name, version = _project_identity(pyproject)
    expected_tag = expected_release_tag(version)
    if tag != expected_tag:
        raise ReleaseValidationError(
            f"Release tag {tag!r} does not match package version {version!r}; "
            f"expected {expected_tag!r}"
        )
    if dist is None:
        if github_output is not None:
            raise ReleaseValidationError("Verified output requires built distributions")
        return version
    wheel, sdist = _distribution_paths(dist, name, version)
    identity = (name, version)
    _validate_wheel(wheel, identity)
    _validate_sdist(sdist, identity)
    if github_output is not None:
        try:
            with github_output.open("a", encoding="utf-8", newline="\n") as output:
                output.write(f"verified=true\npackage_version={version}\n")
        except OSError as exc:
            raise ReleaseValidationError(
                f"Unable to write GitHub output {github_output}: {exc}"
            ) from exc
    return version


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a ClimaDC release tag and artifacts")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--github-output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        version = validate_release(args.tag, args.pyproject, args.dist, args.github_output)
    except ReleaseValidationError as exc:
        raise SystemExit(f"release validation failed: {exc}") from exc
    print(f"release validation passed: {version}")


if __name__ == "__main__":
    main()
