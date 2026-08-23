from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_docs_workflow_builds_on_pr_and_deploys_only_after_main_push() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8"))
    jobs = workflow["jobs"]

    build_steps = jobs["build"]["steps"]
    assert any(step.get("run") == "mkdocs build --strict" for step in build_steps)

    upload = next(
        step
        for step in build_steps
        if step.get("uses")
        == "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9"
    )
    assert upload["if"] == "github.event_name == 'push'"
    assert upload["with"]["path"] == "site"

    deploy = jobs["deploy"]
    assert deploy["if"] == "github.event_name == 'push'"
    assert deploy["needs"] == "build"
    assert deploy["environment"]["name"] == "github-pages"
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    assert any(
        step.get("uses") == "actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128"
        for step in deploy["steps"]
    )
