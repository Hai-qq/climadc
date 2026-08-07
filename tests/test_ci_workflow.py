from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _collect_xarray_tests_with_failed_import(
    tmp_path: Path, *, missing_module: str
) -> subprocess.CompletedProcess[str]:
    (tmp_path / "sitecustomize.py").write_text(
        f"""
import importlib.abc
import sys


class BlockXarray(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "xarray" or fullname.startswith("xarray."):
            raise ModuleNotFoundError(
                "No module named '{missing_module}'", name="{missing_module}"
            )
        return None


sys.meta_path.insert(0, BlockXarray())
""".lstrip(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(tmp_path), env.get("PYTHONPATH", ""))))

    return subprocess.run(
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


def test_xarray_tests_skip_collection_when_optional_dependency_is_missing(
    tmp_path: Path,
) -> None:
    result = _collect_xarray_tests_with_failed_import(tmp_path, missing_module="xarray")
    output = result.stdout + result.stderr
    assert result.returncode in {0, 5}, output
    assert "skipped" in output.lower()
    assert "error" not in output.lower()


def test_xarray_tests_expose_broken_transitive_dependency(tmp_path: Path) -> None:
    result = _collect_xarray_tests_with_failed_import(tmp_path, missing_module="broken_dependency")
    output = result.stdout + result.stderr

    assert result.returncode == 2, output
    assert "broken_dependency" in output
    assert "error collecting" in output.lower()
    assert "skipped" not in output.lower()


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


def test_scheduled_reference_provider_workflow_runs_only_network_contract_test() -> None:
    path = ROOT / ".github" / "workflows" / "network-checks.yml"
    text = path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    steps = workflow["jobs"]["reference-providers"]["steps"]
    commands = [step["run"] for step in steps if "run" in step]

    assert "schedule:" in text
    assert "workflow_dispatch:" in text
    assert "python -m pytest -m network tests/integration/test_reference_network.py -q" in commands
    assert workflow["permissions"] == {"contents": "read"}
