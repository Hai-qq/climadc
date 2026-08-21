# Replay sensitivity and robustness suites

A replay suite runs complete, independently verifiable studies and aggregates signed deltas from
each scenario's own ASAP baseline. `suite_type` fixes what the matrix is allowed to claim:

- `sensitivity` changes assumptions within a shared sample, such as an objective carbon price or
  synthetic demand charge;
- `robustness` requires a declared and actually varying `decision_date`, `season`, `location`, or
  `workload` dimension. Different values alone do not prove statistical independence; provenance
  and coverage still bound the claim.

The packaged suite reuses one London day and workload, so it is sensitivity analysis:

```yaml
schema_version: "1"
suite_id: gb-london-policy-sensitivity
suite_type: sensitivity
aggregation: equal_weight
scenarios:
  - scenario_id: balanced
    description: Illustrative 1000 GBP/tCO2e carbon price
    study: study.yaml
  - scenario_id: cost-dominant
    description: Illustrative 100 GBP/tCO2e carbon price
    study: study-cost-dominant.yaml
```

Run and verify it offline:

```bash
climadc demo sensitivity-suite --output-dir ./climadc-replay-suite-runs
climadc verify-suite ./climadc-replay-suite-runs/latest
climadc report ./climadc-replay-suite-runs/latest
```

The old `demo robustness-suite` spelling calls the same packaged sensitivity suite and emits a
deprecation warning. It is not an out-of-sample validation alias.

## Comparability and aggregation

Scenarios must share horizon, interval, single-window/rolling shape, risk-policy availability,
metric schema, policy order, and currency. Objective values may vary in a sensitivity matrix.
Each policy row records feasibility and signed cost, estimated location-based emissions, and peak
deltas from that scenario's ASAP schedule. Equal weights are arithmetic weights, not probabilities.
The published “worst” value is only the maximum among the finite declared scenarios, not tail risk.

The aggregate Pareto set minimizes the three equal-weight mean deltas among policies feasible in
every scenario. It must not be confused with `objective.mode: pareto_analysis`, which publishes a
complete fixed carbon-price sweep inside one study.

## Artifact contract

The v2 suite root contains `suite.yaml`, lineage and environment manifests, recursive checksums,
`scenario-index.json`, `scenario-metrics.parquet`, `suite-metrics.json`, `pareto-frontier.json`, an
offline report, and `scenarios/`. Every scenario directory is a complete v2 run and is recursively
checked by `verify-suite`. File membership comes from `run-manifest.json`; the list above describes
the current schema rather than promising a permanent count.

The HTML for a sensitivity suite deliberately avoids a robustness-validation claim. A robust
matrix still needs evidence-level limits: different dates with a synthetic workload remain E1, not
E2 or E3.
