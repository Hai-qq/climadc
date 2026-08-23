from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_dependabot_preserves_minimum_compatibility_constraints() -> None:
    config = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text(encoding="utf-8"))
    pip_updates = next(
        update for update in config["updates"] if update["package-ecosystem"] == "pip"
    )

    assert "constraints/minimum.txt" in pip_updates["exclude-paths"]
