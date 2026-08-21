from __future__ import annotations

import json
import math
from collections.abc import Mapping
from html import escape
from numbers import Real
from typing import Any, cast

import pandas as pd

from climadc.replay.suite import ReplaySuiteResult


def _machine_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _machine_value(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_machine_value(nested) for nested in value]
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    if hasattr(value, "item"):
        return _machine_value(cast(Any, value).item())
    return value


def _machine_report_payload(result: ReplaySuiteResult) -> str:
    payload = {
        "schema_version": "1",
        "suite_type": result.config.suite_type,
        "records": result.suite_metrics.to_dict(orient="records"),
        "pareto_policies": list(result.pareto_frontier),
    }
    text = json.dumps(
        _machine_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return escape(text, quote=False)


def _number(value: object, digits: int = 2) -> str:
    checked = cast(Any, value)
    if checked is None or bool(pd.isna(checked)):
        return "—"
    return f"{float(checked):,.{digits}f}"


def _percent(value: object) -> str:
    checked = cast(Any, value)
    if checked is None or bool(pd.isna(checked)):
        return "—"
    return f"{100.0 * float(checked):.1f}%"


def _aggregate_rows(result: ReplaySuiteResult) -> str:
    rows: list[str] = []
    for _, metric in result.suite_metrics.iterrows():
        rows.append(
            "<tr>"
            f"<th>{escape(str(metric['policy']))}</th>"
            f"<td>{int(metric['feasible_scenarios'])}/{int(metric['scenario_count'])}</td>"
            f"<td>{_percent(metric['cost_improvement_fraction_of_feasible'])}</td>"
            f"<td>{_number(metric['mean_energy_cost_change_vs_asap'])}</td>"
            f"<td>{_number(metric['worst_energy_cost_change_vs_asap'])}</td>"
            f"<td>{escape(str(metric['worst_energy_cost_scenario'])) if pd.notna(metric['worst_energy_cost_scenario']) else '—'}</td>"
            f"<td>{_percent(metric['emissions_improvement_fraction_of_feasible'])}</td>"
            f"<td>{_number(metric['mean_estimated_location_based_emissions_change_vs_asap_kgco2e'])}</td>"
            f"<td>{_number(metric['worst_estimated_location_based_emissions_change_vs_asap_kgco2e'])}</td>"
            f"<td>{escape(str(metric['worst_emissions_scenario'])) if pd.notna(metric['worst_emissions_scenario']) else '—'}</td>"
            f"<td>{_percent(metric['peak_improvement_fraction_of_feasible'])}</td>"
            f"<td>{_number(metric['mean_peak_change_vs_asap_kw'])}</td>"
            f"<td>{_number(metric['worst_peak_change_vs_asap_kw'])}</td>"
            f"<td>{escape(str(metric['worst_peak_scenario'])) if pd.notna(metric['worst_peak_scenario']) else '—'}</td>"
            f"<td>{'yes' if bool(metric['pareto_efficient']) else 'no'}</td>"
            "</tr>"
        )
    return "".join(rows)


def _scenario_rows(result: ReplaySuiteResult, scenario_paths: Mapping[str, str]) -> str:
    rows: list[str] = []
    for scenario in result.scenarios:
        feasible = bool(scenario.study.replay.status["feasible"].all())
        relative = scenario_paths[scenario.scenario_id]
        rows.append(
            "<tr>"
            f"<th>{escape(scenario.scenario_id)}</th>"
            f"<td>{escape(scenario.description)}</td>"
            f"<td>{escape(scenario.study.config.study_id)}</td>"
            f"<td>{escape(scenario.study.config.decision_time.isoformat())}</td>"
            f"<td>{'yes' if feasible else 'no'}</td>"
            f"<td><code>{escape(scenario.study.config_sha256[:12])}</code></td>"
            f'<td><a href="{escape(relative)}/report.html">scenario report</a></td>'
            "</tr>"
        )
    return "".join(rows)


def _declared_rows(result: ReplaySuiteResult) -> str:
    values: list[tuple[str, object]] = [
        ("Aggregation", "equal weight per declared scenario"),
        ("Baseline", "ASAP within each scenario"),
        ("Replay mode", result.mode),
        ("Currency", result.currency),
        ("Policies", list(result.policies)),
    ]
    values.extend(
        (key.replace("_", " ").title(), value) for key, value in result.config.assumptions.items()
    )
    return "".join(
        "<tr>"
        f"<th>{escape(label)}</th>"
        f"<td>{escape(value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))}</td>"
        "</tr>"
        for label, value in values
    )


def _limitations(result: ReplaySuiteResult) -> str:
    items = [
        "Equal scenario weights are an analysis convention, not observed probabilities.",
        (
            "Mean and worst signed changes are relative to ASAP inside each scenario; they are "
            "not production savings or guarantees."
        ),
        (
            "Improvement fractions use only scenarios where that policy is feasible; Pareto "
            "eligibility requires feasibility in every scenario."
        ),
        (
            "The Pareto frontier minimizes three scenario-average deltas and is not a global or "
            "probability-weighted optimum."
        ),
        *result.config.limitations,
    ]
    return "".join(f"<li>{escape(item)}</li>" for item in dict.fromkeys(items))


def render_replay_suite_report(
    result: ReplaySuiteResult,
    run_id: str,
    scenario_paths: Mapping[str, str],
) -> str:
    """Render a self-contained cross-scenario sensitivity or robustness report."""

    pareto = ", ".join(result.pareto_frontier) if result.pareto_frontier else "none"
    fully_feasible = int(
        (result.suite_metrics["feasible_scenarios"] == len(result.scenarios)).sum()
    )
    currency = escape(result.currency)
    suite_type = result.config.suite_type
    heading = "SENSITIVITY ANALYSIS" if suite_type == "sensitivity" else "ROBUSTNESS ANALYSIS"
    title = "sensitivity suite" if suite_type == "sensitivity" else "robustness suite"
    policy_heading = "Policy sensitivity" if suite_type == "sensitivity" else "Policy robustness"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ClimaDC replay {title} — {escape(result.config.suite_id)}</title>
  <style>
    :root {{ color-scheme:light; --ink:#17212b; --muted:#5e6872; --line:#d9e0e5;
      --paper:#fff; --wash:#f4f7f8; --accent:#0b6b66; --warn:#925006; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--wash);
      font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ max-width:1480px; margin:0 auto; padding:40px 24px 64px; }}
    header,section {{ background:var(--paper); border:1px solid var(--line); border-radius:12px;
      margin-bottom:18px; padding:24px; }} h1 {{ margin:0 0 8px; font-size:30px; }}
    h2 {{ margin-top:0; font-size:20px; }} p {{ margin:8px 0; }} .muted {{ color:var(--muted); }}
    .badge {{ display:inline-block; border-radius:999px; padding:4px 9px; color:#fff;
      background:var(--accent); font-weight:700; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
    .card {{ border:1px solid var(--line); border-radius:9px; padding:14px; }}
    .card b {{ display:block; font-size:21px; }} .table-wrap {{ overflow-x:auto; }}
    table {{ border-collapse:collapse; width:100%; min-width:900px; }}
    table.aggregate {{ min-width:1600px; }} th,td {{ border-bottom:1px solid var(--line);
      padding:10px 8px; text-align:right; vertical-align:top; white-space:nowrap; }}
    th:first-child,td:first-child {{ text-align:left; }} thead th {{ color:var(--muted);
      font-size:12px; }} table.declared {{ min-width:0; table-layout:fixed; }}
    table.declared th,table.declared td {{ text-align:left; white-space:normal; overflow-wrap:anywhere; }}
    table.declared th {{ width:24%; }} code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
    a {{ color:var(--accent); }} .callout {{ border-left:4px solid var(--warn); padding-left:14px; }}
    ul {{ padding-left:22px; }} @media print {{ body {{ background:#fff; }} main {{ max-width:none;
      padding:0; }} section,header {{ break-inside:avoid; }} }}
  </style>
</head>
<body><main>
  <template id="climadc-report-data">{_machine_report_payload(result)}</template>
  <header>
    <span class="badge">EQUAL-WEIGHT {heading}</span>
    <h1>ClimaDC replay suite</h1>
    <p><strong>{escape(result.config.suite_id)}</strong> · run <code>{escape(run_id)}</code></p>
    <p class="muted">Cross-scenario comparison of verified replay studies. Negative cost,
      emissions, and peak deltas are reductions relative to each scenario's ASAP baseline.</p>
  </header>
  <section>
    <h2>Suite boundary</h2>
    <div class="grid">
      <div class="card"><span>Scenarios</span><b>{len(result.scenarios)}</b></div>
      <div class="card"><span>Policies</span><b>{len(result.policies)}</b></div>
      <div class="card"><span>Fully feasible policies</span><b>{fully_feasible}</b></div>
      <div class="card"><span>Currency</span><b>{currency}</b></div>
      <div class="card"><span>Pareto frontier</span><b>{escape(pareto)}</b></div>
    </div>
    <p class="callout">Scenario means are unweighted arithmetic means. A policy enters the
      Pareto comparison only when it is feasible in every declared scenario.</p>
  </section>
  <section>
    <h2>{policy_heading}</h2>
    <p class="muted">Worst case is the maximum signed delta among feasible scenarios. Improvement
      rate is the share of feasible scenarios whose delta is below -1e-9 in the published unit.</p>
    <div class="table-wrap"><table class="aggregate">
      <thead><tr><th>Policy</th><th>Feasible</th>
      <th>Cost improve</th><th>Mean Δ cost ({currency})</th><th>Worst Δ cost</th><th>Worst scenario</th>
      <th>CO₂ improve</th><th>Mean Δ kgCO₂e</th><th>Worst Δ kgCO₂e</th><th>Worst scenario</th>
      <th>Peak improve</th><th>Mean Δ peak kW</th><th>Worst Δ peak kW</th><th>Worst scenario</th>
      <th>Pareto</th></tr></thead><tbody>{_aggregate_rows(result)}</tbody>
    </table></div>
  </section>
  <section>
    <h2>Scenario lineage</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Scenario</th><th>Description</th><th>Study</th><th>Decision time</th>
      <th>Feasible</th><th>Config hash</th><th>Evidence</th></tr></thead>
      <tbody>{_scenario_rows(result, scenario_paths)}</tbody>
    </table></div>
  </section>
  <section>
    <h2>Declared assumptions</h2>
    <table class="declared"><tbody>{_declared_rows(result)}</tbody></table>
  </section>
  <section>
    <h2>Limitations</h2>
    <ul>{_limitations(result)}</ul>
  </section>
</main></body></html>
"""


__all__ = ["render_replay_suite_report"]
