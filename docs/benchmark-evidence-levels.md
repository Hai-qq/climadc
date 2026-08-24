# Benchmark evidence levels

The level describes the provenance and validation design, not whether a run completed successfully.
Higher levels require new evidence; they cannot be obtained by relabelling a lower-level fixture.

| Level | Minimum evidence | Permitted interpretation | Current ClimaDC evidence |
|---|---|---|---|
| **E0** | Project-generated synthetic signals/workload used to exercise contracts and plumbing | Synthetic pipeline sanity check, deterministic regression, or mechanism example | WeatherDC small fixture |
| **E1** | External grid/weather signals with a synthetic workload and explicitly declared facility/tariff assumptions | Signal-grounded mechanism demonstration and accounting check | London 24-hour replay |
| **E2** | Trace-driven causal benchmark with documented forecast vintages, workload trace conversion, mapping assumptions, and independent dates/sites/workloads | Out-of-sample benchmark within the trace and assumptions | None; `DATA_REQUIRED` |
| **E3** | Same-site, same-period operational signals, workload/control logs, measured outcomes, and a defensible counterfactual or experimental design | Operational validation within the stated site/period/design | None; `DATA_REQUIRED` |

E0 and E1 can verify software behavior and expose trade-offs. They cannot establish deployment
readiness, production savings, marginal avoided emissions, generalization, or causal effects at a
real site. A same-day objective-weight sweep is sensitivity analysis, not robustness validation.

## Advancement gates

An E2 package needs immutable trace licensing and hashes, prospective or historically archived
forecast vintages, causal availability timestamps, explicit deadline/slack/preemptibility and power
mapping assumptions, independent evaluation slices, and the complete v2 verifier contract. Google
Borg or Alibaba Cluster Trace may supply workload traces, but neither becomes a same-site history
when combined with unrelated UK grid/weather data.

The Google ClusterData2019 converter covers only the trace-conversion and provenance portion of
that gate. Its manifests remain `DATA_REQUIRED` and claim-ineligible; see
[trace-driven benchmark preparation](concepts/trace-driven-benchmarks.md).

E3 additionally needs permissioned same-site telemetry and workload records, measured data quality,
documented interventions or a credible counterfactual design, operational safety review, and an
analysis plan fixed before outcome selection. No current repository artifact meets those gates.
