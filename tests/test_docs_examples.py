from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


_QUICKSTART_FENCE = re.compile(
    r"```bash quickstart-test\n(?P<script>.*?)\n```",
    flags=re.DOTALL,
)
_EXPECTED_ARTIFACTS = {
    "run.yaml",
    "lineage.json",
    "splits.parquet",
    "predictions.parquet",
    "metrics.json",
    "leakage-report.json",
    "dataset-card.md",
    "report.html",
    "run-manifest.json",
    "environment.json",
    "checksums.sha256",
}


def test_documented_quickstart_runs_offline_and_publishes_verifiable_v2_artifacts(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("The tested Quickstart block is POSIX; docs include a PowerShell equivalent")
    docs_path = Path(__file__).parents[1] / "docs" / "quickstart.md"
    text = docs_path.read_text(encoding="utf-8")
    match = _QUICKSTART_FENCE.search(text)
    assert match is not None, "docs/quickstart.md must contain a quickstart-test bash fence"

    script = match.group("script")
    commands = [line for line in script.splitlines() if line.strip()]
    assert len(commands) == 6
    assert sum(command.lstrip().startswith("climadc ") for command in commands) == 5

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("The tested Quickstart block requires a POSIX-compatible bash")
    environment = os.environ.copy()
    environment["TMPDIR"] = str(tmp_path)
    completed = subprocess.run(
        [bash, "-euo", "pipefail", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    report_path = Path(completed.stdout.strip().splitlines()[-1])
    run_dir = report_path.parent
    assert {path.name for path in run_dir.iterdir()} == _EXPECTED_ARTIFACTS
    assert report_path.is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "leakage-report.json").is_file()
