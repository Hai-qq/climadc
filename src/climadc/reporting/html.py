from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from climadc.benchmark import RunResult

_TEMPLATES = Path(__file__).with_name("templates")


def render_report(result: RunResult, run_id: str) -> str:
    """Render one self-contained, autoescaped benchmark report."""

    environment = Environment(
        loader=FileSystemLoader(_TEMPLATES),
        autoescape=select_autoescape(enabled_extensions=("html", "j2"), default=True),
        keep_trailing_newline=True,
    )
    template = environment.get_template("report.html.j2")
    prediction_rows = len(result.predictions.to_pandas())
    split_ids = sorted(str(value) for value in result.splits["split_id"].unique())
    model_ids = sorted(str(value) for value in result.predictions.to_pandas()["model_id"].unique())
    decision_state = (
        "disabled"
        if result.decision is None
        else "feasible"
        if result.decision.feasible
        else "infeasible"
    )
    return template.render(
        run_id=run_id,
        study_id=result.study_id,
        started_at=result.started_at.isoformat(),
        config_sha256=result.config_sha256,
        prediction_rows=prediction_rows,
        split_ids=split_ids,
        model_ids=model_ids,
        accepted_rows=result.leakage_audit.accepted_rows,
        rejected_rows=result.leakage_audit.rejected_rows,
        decision_state=decision_state,
        metrics_json=json.dumps(result.metrics, sort_keys=True, indent=2, allow_nan=False),
    )
