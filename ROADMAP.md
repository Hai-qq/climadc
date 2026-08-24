# ClimaDC roadmap

This roadmap preserves ClimaDC's boundary as an offline evaluation and replay framework.

## v0.3 — verifiable evidence release

- Versioned artifact schema v2 and independent run/suite verification.
- Raw-response lineage for new Open-Meteo/NESO refreshes.
- Evidence levels and hash-bound claims.
- Dimensionally explicit objectives and correctly named sensitivity analysis.
- Compact, code-generated E1 reference summary and supply-chain hardening.

## Benchmark v1 — DATA_REQUIRED

Prospective forecast-vintage capture and trace conversion may be implemented before results exist.
Publishing E2 results is blocked until the repository has licensed immutable workload traces,
historically valid forecast vintages, causal availability facts, independent evaluation slices, and
documented deadline/slack/preemptibility/power mappings. Google Borg or Alibaba Cluster Trace may
be supported as foreign workload traces; neither may be described as same-site London history.

Current foundation: a bounded Google ClusterData2019 v3 CSV converter and independent verifier are
implemented. They bind source bytes, exact export SQL, mapping config, canonical workload, and
artifact membership. No public trace export has been acquired, no forecast-vintage capture has run,
and no E2 replay or claim exists. The remaining gates are tracked in the
[trace-driven benchmark guide](docs/concepts/trace-driven-benchmarks.md).

## Operational validation — DATA_REQUIRED

E3 requires permissioned same-site, same-period telemetry and workload/control logs, measured data
quality, an approved safety boundary, and a defensible causal or counterfactual analysis plan. No
online actuation, Kubernetes scheduling, RL, dashboard, digital-twin, or model-zoo work is planned.
