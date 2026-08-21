from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from climadc.errors import ConfigurationError
from climadc.evidence.checksums import (
    load_checksums,
    safe_relative_path,
    verify_checksums,
    write_checksums,
)


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "..",
        "../escape",
        "nested/../escape",
        "/absolute/file",
        "C:/absolute/file",
        "C:\\absolute\\file",
        "nested\\windows.txt",
    ],
)
def test_safe_relative_path_rejects_ambiguous_or_escaping_paths(value: str) -> None:
    with pytest.raises(ConfigurationError):
        safe_relative_path(value)


@given(st.builds(lambda prefix, suffix: f"{prefix}\\{suffix}", st.text(), st.text()))
def test_safe_relative_path_property_rejects_every_backslash_path(value: str) -> None:
    with pytest.raises(ConfigurationError):
        safe_relative_path(value)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_checksum_parser_accepts_lf_and_crlf(newline: str, tmp_path: Path) -> None:
    digest = hashlib.sha256(b"payload").hexdigest()
    checksum = tmp_path / "checksums.sha256"
    checksum.write_bytes(f"{digest}  nested/file.txt{newline}".encode())

    assert load_checksums(checksum) == {"nested/file.txt": digest}


def test_verify_checksums_detects_extra_missing_and_modified_files(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("original\n", encoding="utf-8")
    write_checksums(tmp_path)
    assert set(verify_checksums(tmp_path)) == {"artifact.txt"}

    artifact.write_text("modified\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Checksum mismatch"):
        verify_checksums(tmp_path)

    artifact.write_text("original\n", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("undeclared\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="extra"):
        verify_checksums(tmp_path)

    (tmp_path / "extra.txt").unlink()
    artifact.unlink()
    with pytest.raises(ConfigurationError, match="missing"):
        verify_checksums(tmp_path)


def test_checksum_parser_rejects_duplicate_unsorted_and_self_entries(tmp_path: Path) -> None:
    digest = "0" * 64
    checksum = tmp_path / "checksums.sha256"
    for text, message in (
        (f"{digest}  b\n{digest}  a\n", "stable lexical ordering"),
        (f"{digest}  a\n{digest}  a\n", "Duplicate"),
        (f"{digest}  checksums.sha256\n", "must not hash itself"),
    ):
        checksum.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigurationError, match=message):
            load_checksums(checksum)
