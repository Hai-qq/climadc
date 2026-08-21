from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path, PurePosixPath

from climadc.errors import ConfigurationError

CHECKSUM_FILE = "checksums.sha256"
_CHECKSUM_LINE = re.compile(r"^(?P<digest>[0-9a-f]{64})  (?P<path>[^\r\n]+)$")


def sha256_file(path: Path) -> str:
    """Hash one regular file without following a caller-supplied directory tree."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConfigurationError(f"Unable to hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def safe_relative_path(value: str) -> str:
    """Return a portable POSIX relative path or reject traversal/platform ambiguity."""

    if not isinstance(value, str) or not value or "\\" in value:
        raise ConfigurationError(f"Artifact path must use non-empty POSIX syntax: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ConfigurationError(f"Artifact path must not be absolute or traverse: {value!r}")
    if ":" in path.parts[0]:
        raise ConfigurationError(f"Artifact path must not contain a drive prefix: {value!r}")
    return path.as_posix()


def _is_reparse_point(status: os.stat_result) -> bool:
    attributes = int(getattr(status, "st_file_attributes", 0))
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & flag)


def artifact_files(root: Path, *, exclude: frozenset[str] = frozenset()) -> tuple[str, ...]:
    """Enumerate regular artifact files with stable paths and no symlink traversal."""

    base = Path(root)
    if not base.is_dir():
        raise ConfigurationError(f"Artifact directory does not exist: {base}")
    files: list[str] = []
    for path in sorted(base.rglob("*"), key=lambda item: item.as_posix()):
        try:
            status = path.lstat()
        except OSError as exc:
            raise ConfigurationError(f"Unable to inspect artifact path {path}: {exc}") from exc
        relative = path.relative_to(base).as_posix()
        if relative in exclude:
            continue
        if stat.S_ISLNK(status.st_mode) or _is_reparse_point(status):
            raise ConfigurationError(f"Artifact tree must not contain links: {relative}")
        if stat.S_ISDIR(status.st_mode):
            continue
        if not stat.S_ISREG(status.st_mode):
            raise ConfigurationError(f"Artifact tree contains a non-regular file: {relative}")
        safe_relative_path(relative)
        files.append(relative)
    return tuple(files)


def checksum_text(root: Path) -> str:
    """Build the canonical checksum document for every file except the document itself."""

    base = Path(root)
    names = artifact_files(base, exclude=frozenset({CHECKSUM_FILE}))
    return "".join(f"{sha256_file(base / PurePosixPath(name))}  {name}\n" for name in names)


def write_checksums(root: Path) -> Path:
    path = Path(root) / CHECKSUM_FILE
    path.write_text(checksum_text(root), encoding="utf-8", newline="\n")
    return path


def load_checksums(path: Path) -> dict[str, str]:
    """Parse LF or CRLF input while enforcing canonical ordering and path safety."""

    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"Unable to read checksum file {path}: {exc}") from exc
    records: dict[str, str] = {}
    ordered: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ConfigurationError(f"Invalid checksums.sha256 line {number}: {line!r}")
        name = safe_relative_path(match.group("path"))
        if name == CHECKSUM_FILE:
            raise ConfigurationError("checksums.sha256 must not hash itself")
        if name in records:
            raise ConfigurationError(f"Duplicate checksum path: {name}")
        records[name] = match.group("digest")
        ordered.append(name)
    if not records:
        raise ConfigurationError("checksums.sha256 must contain at least one artifact")
    if ordered != sorted(ordered):
        raise ConfigurationError("checksums.sha256 paths must use stable lexical ordering")
    return records


def verify_checksums(root: Path) -> dict[str, str]:
    """Verify the checksum set, every digest, and the absence of undeclared files."""

    base = Path(root)
    expected = load_checksums(base / CHECKSUM_FILE)
    actual_names = set(artifact_files(base, exclude=frozenset({CHECKSUM_FILE})))
    expected_names = set(expected)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise ConfigurationError(
            f"Checksum artifact set mismatch: missing={missing}, extra={extra}"
        )
    for name, digest in expected.items():
        actual = sha256_file(base / PurePosixPath(name))
        if actual != digest:
            raise ConfigurationError(
                f"Checksum mismatch for {name}: expected {digest}, found {actual}"
            )
    return expected


__all__ = [
    "CHECKSUM_FILE",
    "artifact_files",
    "checksum_text",
    "load_checksums",
    "safe_relative_path",
    "sha256_file",
    "verify_checksums",
    "write_checksums",
]
