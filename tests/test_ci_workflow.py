from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_xarray_tests_skip_collection_when_optional_dependency_is_missing(
    tmp_path: Path,
) -> None:
    (tmp_path / "sitecustomize.py").write_text(
        """
import importlib.abc
import sys


class BlockXarray(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "xarray" or fullname.startswith("xarray."):
            raise ModuleNotFoundError("No module named 'xarray'", name="xarray")
        return None


sys.meta_path.insert(0, BlockXarray())
""".lstrip(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(tmp_path), env.get("PYTHONPATH", ""))))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "tests/adapters/test_xarray.py",
            "-q",
            "-rs",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode in {0, 5}, output
    assert "skipped" in output.lower()
    assert "error" not in output.lower()


def test_quality_job_installs_optional_xarray_with_ci_numpy_constraint() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    quality_commands = [step["run"] for step in jobs["quality"]["steps"] if "run" in step]

    assert 'python -m pip install -e ".[dev,xarray]" "numpy<2.4"' in quality_commands

    for job_name in ("test-ubuntu", "test-cross-platform"):
        commands = [step["run"] for step in jobs[job_name]["steps"] if "run" in step]
        assert 'python -m pip install -e ".[dev]"' in commands

    optional_commands = [
        step["run"] for step in jobs["optional-dependencies"]["steps"] if "run" in step
    ]
    assert 'python -m pip install -e ".[lightgbm,xarray,dev]"' in optional_commands
    assert any("tests/adapters/test_xarray.py" in command for command in optional_commands)
