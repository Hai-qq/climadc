from __future__ import annotations

import json
from html import escape
from typing import Any, cast

from climadc.replay.study import ReplayStudyResult
from climadc.replay.rolling import RollingReplayResult

_LABELS = {
    "asap": "ASAP baseline",
    "peak": "Peak only",
    "price": "Price only",
    "carbon": "Carbon only",
    "joint": "Forecast joint",
    "risk_aware": "Upper-quantile joint",
    "oracle": "Realized oracle",
}
_RISK_SIGNAL_LABELS = {
    "temperature": "Temperature",
    "energy_price": "Energy price",
    "carbon_intensity": "Carbon intensity",
}


def _policy_label(policy: str) -> str:
    if policy.startswith("pareto_cp_"):
        return f"Pareto point ({policy.removeprefix('pareto_cp_').replace('p', '.')} /tCO₂e)"
    return _LABELS.get(policy, policy)


def _number(value: object, digits: int = 2) -> str:
    return f"{float(cast(Any, value)):,.{digits}f}"


def _policy_rows(result: ReplayStudyResult) -> str:
    if result.replay.metrics.empty:
        return '<tr><td colspan="10">No settled metrics: replay infeasible.</td></tr>'
    rows: list[str] = []
    for _, metric in result.replay.metrics.iterrows():
        policy = str(metric["policy"])
        cost_change = float(metric["energy_cost_change_vs_asap"])
        emissions_change = float(metric["estimated_location_based_emissions_change_vs_asap_kgco2e"])
        peak_change = float(metric["peak_change_vs_asap_kw"])
        change_class = "neutral" if policy == "asap" else ""
        rows.append(
            "<tr>"
            f"<th>{escape(_policy_label(policy))}</th>"
            f"<td>{_number(metric['facility_energy_kwh'])}</td>"
            f"<td>{_number(metric['cooling_energy_kwh'])}</td>"
            f"<td>{_number(metric['estimated_location_based_emissions_kgco2e'])}</td>"
            f"<td>{_number(metric['energy_cost'])}</td>"
            f"<td>{_number(metric['peak_kw'])}</td>"
            f'<td class="{change_class}">{_number(cost_change)}</td>'
            f'<td class="{change_class}">{_number(emissions_change)}</td>'
            f'<td class="{change_class}">{_number(peak_change)}</td>'
            f"<td>{_number(metric['objective_regret'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def _source_rows(result: ReplayStudyResult) -> str:
    rows: list[str] = []
    for source in result.manifest.records:
        rows.append(
            "<tr>"
            f"<th>{escape(source.source_id)}</th>"
            f"<td>{escape(source.provider)}</td>"
            f"<td>{escape(source.role)}</td>"
            f"<td>{escape(source.provenance)}</td>"
            f"<td>{escape(source.license)}</td>"
            f"<td><code>{escape(source.sha256[:12])}</code></td>"
            "</tr>"
        )
    return "".join(rows)


def _limitations(result: ReplayStudyResult) -> str:
    items = list(result.config.limitations)
    if not items:
        for source in result.manifest.records:
            items.extend(source.limitations)
    deduplicated = list(dict.fromkeys(items))
    return "".join(f"<li>{escape(item)}</li>" for item in deduplicated)


def _forecast_summary(result: ReplayStudyResult) -> str:
    metrics = result.forecast_metrics
    if metrics["status"] != "computed":
        return escape(str(metrics.get("reason", "not computed")))
    interval_note = (
        f"Risk policy uses the declared q={result.config.replay.risk_quantile:g} scenario; "
        "committed-slot marginal diagnostics are reported below, while two-sided interval "
        "coverage remains unavailable."
        if result.config.replay.risk_quantile is not None
        else "Interval coverage is not applicable because the study contains point forecasts only."
    )
    return (
        f"Temperature MAE: <strong>{_number(metrics['temperature_mae_c'])} °C</strong>; "
        "carbon-intensity MAE: "
        f"<strong>{_number(metrics['carbon_intensity_mae_gco2e_per_kwh'])} "
        "gCO₂e/kWh</strong>; scenario-tariff MAE: "
        f"<strong>{_number(metrics['energy_price_mae_per_kwh'], 4)} "
        f"{escape(result.replay.currency)}/kWh</strong>. {interval_note}"
    )


def _risk_diagnostics_section(result: ReplayStudyResult) -> str:
    status = str(result.forecast_metrics.get("upper_quantile_diagnostics_status", "not_configured"))
    if status == "not_configured":
        return ""
    diagnostics_value = result.forecast_metrics.get("upper_quantile_diagnostics")
    if status != "computed" or diagnostics_value is None:
        return (
            '<section><h2>Upper-quantile diagnostics</h2><p class="muted">'
            "Not computed because the replay did not produce settlement profiles."
            "</p></section>"
        )
    diagnostics = cast(dict[str, Any], diagnostics_value)
    signals = cast(dict[str, dict[str, Any]], diagnostics["signals"])
    rows: list[str] = []
    for name in ("temperature", "energy_price", "carbon_intensity"):
        signal = signals[name]
        rows.append(
            "<tr>"
            f"<th>{escape(_RISK_SIGNAL_LABELS[name])}</th>"
            f"<td>{escape(str(signal['unit']))}</td>"
            f"<td>{int(signal['sample_count'])}</td>"
            f"<td>{_number(100.0 * float(signal['empirical_coverage']), 1)}%</td>"
            f"<td>[{_number(100.0 * float(signal['wilson_95_lower']), 1)}%, "
            f"{_number(100.0 * float(signal['wilson_95_upper']), 1)}%]</td>"
            f"<td>{_number(100.0 * float(signal['coverage_gap']), 1)} pp</td>"
            f"<td>{int(signal['exceedance_count'])}</td>"
            f"<td>{_number(signal['mean_positive_exceedance'], 4)}</td>"
            f"<td>{_number(signal['mean_exceedance_when_exceeded'], 4)}</td>"
            f"<td>{_number(signal['maximum_exceedance'], 4)}</td>"
            f"<td>{_number(signal['pinball_loss'], 4)}</td>"
            "</tr>"
        )
    quantile = float(diagnostics["nominal_quantile"])
    confidence = float(diagnostics["confidence_level"])
    return (
        "<section><h2>Upper-quantile diagnostics</h2>"
        f'<p class="muted">Post-hoc marginal backtest over {int(diagnostics["sample_count"])} '
        f"committed slots at q={quantile:g}. Coverage counts actual values at or below the "
        f"declared quantile; the interval is a {confidence:.0%} Wilson binomial interval. "
        "These diagnostics neither recalibrate the schedule nor establish joint coverage, CVaR, "
        "or a production guarantee.</p>"
        '<div class="table-wrap"><table class="metrics">'
        "<thead><tr><th>Signal</th><th>Unit</th><th>n</th><th>Coverage</th>"
        "<th>Wilson interval</th><th>Gap vs q</th><th>Exceedances</th>"
        "<th>Mean positive exceedance</th><th>Mean when exceeded</th>"
        "<th>Maximum exceedance</th><th>Pinball loss</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def _display(value: object) -> str:
    if isinstance(value, str):
        return escape(value)
    return escape(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))


def _assumption_rows(result: ReplayStudyResult) -> str:
    replay = result.config.replay
    model = result.config.facility_model
    rows: list[tuple[str, object]] = [
        ("Horizon / interval", f"{replay.horizon} / {replay.interval}"),
        ("IT capacity / fixed IT", f"{replay.it_capacity_kw} kW / {replay.fixed_it_power_kw} kW"),
        ("Objective contract", replay.objective_payload()),
        ("Risk quantile", replay.risk_quantile if replay.risk_quantile is not None else "disabled"),
        (
            "Facility model",
            (
                f"temperature-sensitive PUE: base={model.base_pue}, "
                f"reference={model.reference_temperature_c} °C, "
                f"slope={model.slope_per_degree_c}/°C, "
                f"bounds=[{model.min_pue}, {model.max_pue}]"
            ),
        ),
    ]
    if result.config.rolling is not None:
        rows.append(
            (
                "Rolling decisions / commit step",
                f"{result.config.rolling.periods} / {result.config.rolling.step}",
            )
        )
    rows.extend(
        (key.replace("_", " ").title(), value) for key, value in result.config.assumptions.items()
    )
    return "".join(
        f"<tr><th>{escape(label)}</th><td>{_display(value)}</td></tr>" for label, value in rows
    )


def _solver_rows(result: ReplayStudyResult) -> str:
    return "".join(
        "<tr>"
        f"<th>{escape(_policy_label(str(row['policy'])))}</th>"
        f"<td>{'yes' if bool(row['feasible']) else 'no'}</td>"
        f"<td>{escape(str(row['solver_status']))}</td>"
        f"<td>{escape(str(row['message']))}</td>"
        "</tr>"
        for _, row in result.replay.status.iterrows()
    )


def _violation_summary(result: ReplayStudyResult) -> str:
    violations = [
        f"{policy}: {message}"
        for policy, messages in result.replay.violations.items()
        for message in messages
    ]
    if not violations:
        return "<p>No replay constraint violations were reported.</p>"
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in violations) + "</ul>"


def _machine_report_payload(result: ReplayStudyResult) -> str:
    payload = {
        "schema_version": "1",
        "objective": result.config.replay.objective_payload(),
        "policies": result.replay.metrics.to_dict(orient="records"),
    }
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda value: cast(Any, value).item(),
    )
    return escape(text, quote=False)


def _objective_note(result: ReplayStudyResult) -> str:
    payload = result.config.replay.objective_payload()
    if payload["mode"] == "legacy_unscaled":
        return (
            "The legacy score is dimensionally unscaled, deprecated, and is neither a currency "
            "value nor a unified utility improvement."
        )
    if payload["mode"] == "epsilon_constraint":
        return "The objective is monetary cost subject to the declared decision-basis bounds."
    if payload["mode"] == "pareto_analysis":
        return (
            "Every declared carbon-price point is shown; no single preferred point is selected. "
            "Objective and regret values use the currency shown above."
        )
    return "The monetized objective and regret values use the currency shown above."


def _pareto_section(result: ReplayStudyResult) -> str:
    objective = result.config.replay.objective
    if objective is None or objective.mode != "pareto_analysis":
        return ""
    rows = []
    metrics = result.replay.metrics.set_index("policy")
    for price in objective.carbon_prices_currency_per_tco2e:
        policy = f"pareto_cp_{format(price, '.12g').replace('.', 'p')}"
        metric = metrics.loc[policy]
        rows.append(
            "<tr>"
            f"<th>{_number(price, 2)}</th>"
            f"<td>{_number(metric['energy_cost'])}</td>"
            f"<td>{_number(metric['estimated_location_based_emissions_kgco2e'])}</td>"
            f"<td>{_number(metric['peak_kw'])}</td>"
            "</tr>"
        )
    return (
        "<section><h2>Declared multi-point analysis</h2>"
        '<p class="muted">Complete fixed carbon-price sweep; rows are not filtered by outcome.</p>'
        '<div class="table-wrap"><table class="metrics">'
        "<thead><tr><th>Carbon price /tCO₂e</th><th>Cost</th><th>Estimated location-based "
        "kgCO₂e</th><th>Peak kW</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def render_replay_report(result: ReplayStudyResult, run_id: str) -> str:
    """Render an offline report with inline CSS and no scripts or external assets."""

    feasible = bool(result.replay.status["feasible"].all())
    status = "FEASIBLE" if feasible else "INFEASIBLE"
    currency = escape(result.replay.currency)
    if isinstance(result.replay, RollingReplayResult):
        timing_summary = (
            f"{result.replay.decision_count} rolling decisions from "
            f"{result.config.decision_time.isoformat()}, committing "
            f"{result.replay.commit_interval} each"
        )
        decision_count = result.replay.decision_count
        oracle_heading = "Oracle Δ"
        oracle_note = (
            " The rolling oracle sees realized signals at each origin but only causally available "
            "jobs, so this signed cumulative difference can be negative after future arrivals."
        )
    else:
        timing_summary = f"Decision time {result.config.decision_time.isoformat()}"
        decision_count = 1
        oracle_heading = "Oracle regret"
        oracle_note = ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ClimaDC engineering replay — {escape(result.config.study_id)}</title>
  <style>
    :root {{ color-scheme: light; --ink:#15202b; --muted:#5b6670; --line:#d8dee4;
      --paper:#ffffff; --wash:#f5f7f9; --accent:#116466; --warning:#8a4b08; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      color:var(--ink); background:var(--wash); }}
    main {{ max-width:1180px; margin:0 auto; padding:40px 24px 64px; }}
    header, section {{ background:var(--paper); border:1px solid var(--line); border-radius:12px;
      padding:24px; margin-bottom:18px; }}
    h1 {{ margin:0 0 8px; font-size:30px; }} h2 {{ margin-top:0; font-size:20px; }}
    p {{ margin:8px 0; }} .muted {{ color:var(--muted); }}
    .badge {{ display:inline-block; padding:4px 9px; border-radius:999px; font-weight:700;
      color:#fff; background:{"#116466" if feasible else "#8a4b08"}; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
    .card {{ border:1px solid var(--line); border-radius:9px; padding:14px; }}
    .card b {{ display:block; font-size:21px; }}
    .table-wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; min-width:860px; }}
    th,td {{ border-bottom:1px solid var(--line); text-align:right; padding:10px 8px;
      vertical-align:top; }}
    th:first-child,td:first-child {{ text-align:left; }} thead th {{ color:var(--muted); font-size:12px; }}
    table.metrics th, table.metrics td {{ white-space:nowrap; }}
    table.lineage {{ min-width:0; table-layout:fixed; }}
    table.lineage th, table.lineage td {{ overflow-wrap:anywhere; text-align:left; }}
    table.lineage th:nth-child(1) {{ width:19%; }} table.lineage th:nth-child(2) {{ width:15%; }}
    table.lineage th:nth-child(3) {{ width:25%; }} table.lineage th:nth-child(4) {{ width:14%; }}
    table.lineage th:nth-child(5) {{ width:12%; }} table.lineage th:nth-child(6) {{ width:15%; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .callout {{ border-left:4px solid var(--warning); padding-left:14px; }}
    ul {{ padding-left:22px; }}
    @media print {{ body {{ background:#fff; }} main {{ max-width:none; padding:0; }} section,header {{ break-inside:avoid; }} }}
  </style>
</head>
<body><main>
  <template id="climadc-report-data">{_machine_report_payload(result)}</template>
  <header>
    <span class="badge">{status}</span>
    <h1>ClimaDC engineering replay</h1>
    <p><strong>{escape(result.config.study_id)}</strong> · run <code>{escape(run_id)}</code></p>
    <p class="muted">{escape(timing_summary)}; all candidate schedules use declared forecasts,
      while settlement and the oracle use post-horizon values.</p>
  </header>
  <section>
    <h2>Study boundary</h2>
    <div class="grid">
      <div class="card"><span>Accepted jobs</span><b>{result.replay.accepted_jobs}</b></div>
      <div class="card"><span>Future jobs excluded</span><b>{result.replay.future_jobs}</b></div>
      <div class="card"><span>Currency</span><b>{currency}</b></div>
      <div class="card"><span>Policies</span><b>{len(result.replay.status)}</b></div>
      <div class="card"><span>Decision windows</span><b>{decision_count}</b></div>
    </div>
    <p>{_forecast_summary(result)}</p>
  </section>
    {_risk_diagnostics_section(result)}
  <section>
    <h2>Declared assumptions</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Field</th><th>Declared value</th></tr></thead>
      <tbody>{_assumption_rows(result)}</tbody>
    </table></div>
  </section>
  <section>
    <h2>Realized settlement by policy</h2>
    <p class="muted">Changes are signed relative to ASAP: negative cost, emissions, or peak values
      are reductions; positive values are increases. {_objective_note(result)}{oracle_note}</p>
    <div class="table-wrap"><table class="metrics">
      <thead><tr><th>Policy</th><th>Facility kWh</th><th>Cooling kWh</th><th>kgCO₂e</th>
      <th>Cost ({currency})</th><th>Peak kW</th><th>Δ cost</th><th>Δ kgCO₂e</th>
      <th>Δ peak kW</th><th>{oracle_heading}</th></tr></thead>
      <tbody>{_policy_rows(result)}</tbody>
    </table></div>
  </section>
  {_pareto_section(result)}
  <section>
    <h2>Solver and constraints</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Policy</th><th>Feasible</th><th>Status</th><th>Message</th></tr></thead>
      <tbody>{_solver_rows(result)}</tbody>
    </table></div>
    {_violation_summary(result)}
  </section>
  <section>
    <h2>Source lineage</h2>
    <div class="table-wrap"><table class="lineage">
      <thead><tr><th>Source ID</th><th>Provider</th><th>Role</th><th>Provenance</th>
      <th>License</th><th>Original SHA-256</th></tr></thead>
      <tbody>{_source_rows(result)}</tbody>
    </table></div>
    <p class="muted">The table shows original verified-input hash prefixes. Full URLs, retrieval
      timestamps, transformations, timing bases, and attribution are preserved in
      <code>source-manifest.yaml</code>, whose complete hashes bind the run-local Parquet files.
      Original complete hashes remain in <code>lineage.json</code>.</p>
  </section>
  <section class="callout">
    <h2>Limitations and claim boundary</h2>
    <ul>{_limitations(result)}</ul>
    <p>This is a deterministic counterfactual replay. It does not actuate infrastructure and must
      not be presented as measured production savings.</p>
  </section>
</main></body></html>
"""
