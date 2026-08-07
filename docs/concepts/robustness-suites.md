# Replay robustness suites

A replay suite turns several independently reproducible replay studies into one cross-scenario
policy comparison. It reuses the existing solver and 12-artifact study contract; it does not mutate
inputs, invent uncertainty distributions, or pool records before scheduling.

## Configure a suite

Each scenario points to a complete replay `study.yaml` with its own source manifest, assumptions,
and input hashes:

```yaml
schema_version: "1"
suite_id: facility-policy-sensitivity
aggregation: equal_weight
scenarios:
  - scenario_id: base
    description: Declared base tariff and objective weights
    study: scenarios/base/study.yaml
  - scenario_id: high-demand-charge
    description: Same horizon with a declared peak-charge stress
    study: scenarios/high-demand-charge/study.yaml
assumptions:
  purpose: Test policy sensitivity to declared operating assumptions
limitations:
  - Scenario weights are not probabilities.
output_dir: replay-suite-runs
```

A suite requires at least two distinct study files and filesystem-safe, unique scenario IDs. Before
solving, ClimaDC requires every scenario to share the replay horizon, interval, single-window versus
rolling mode, rolling shape, and availability of the optional risk-aware policy. After solving, it
also requires one currency and the same ordered policy set. Decision dates, sites, inputs, objective
weights, facility models, and declared tariffs may vary; the author remains responsible for making
their signed deltas scientifically comparable.

Run a custom suite and resolve its report with:

```bash
climadc replay-suite ./suite.yaml --output-dir ./replay-suite-runs
climadc report ./replay-suite-runs/latest
```

The public Python path uses the same implementation:

```python
from pathlib import Path

from climadc.replay import (
    ReplaySuiteArtifactWriter,
    ReplaySuiteConfig,
    ReplaySuiteRunner,
)

config = ReplaySuiteConfig.from_yaml(Path("suite.yaml"))
result = ReplaySuiteRunner().run(config)
run_path = ReplaySuiteArtifactWriter().write(result, config.output_dir)
print(run_path)
```

The installed package includes a fully offline four-scenario sensitivity example:

```bash
climadc demo robustness-suite --output-dir ./climadc-replay-suite-runs
climadc report ./climadc-replay-suite-runs/latest
```

The example reuses one verified Great Britain snapshot and varies joint-objective weights and a
synthetic demand charge. It demonstrates the suite mechanics; it is sensitivity analysis, not an
out-of-sample study.

## Aggregation semantics

Every cost, emissions, and peak value is first expressed as a signed change from ASAP inside its
own scenario. Negative means a reduction; positive means an increase. For each policy the suite
reports:

- scenario count, feasible count, and feasible fraction;
- the fraction of feasible scenarios whose signed change is below `-1e-9` in that metric's unit;
- the unweighted arithmetic mean change over feasible scenarios;
- the worst feasible change, defined as the maximum signed change, plus its scenario ID;
- Pareto membership over mean cost, emissions, and peak changes.

An infeasible policy remains visible but contributes no numeric delta to the mean or improvement
rate. It is never Pareto-eligible. The Pareto comparison includes only policies feasible in every
declared scenario and minimizes all three equal-weight arithmetic means, using a relative numerical
dominance tolerance of `1e-9`. It is not a scalar ranking,
global optimum, probability-weighted risk measure, CVaR calculation, or production guarantee.

## Published evidence

Each immutable suite run contains exactly eight top-level entries:

| Artifact | Purpose |
|---|---|
| `suite.yaml` | path-independent aggregation, compatibility, scenario hashes, assumptions, and limitations |
| `lineage.json` | suite run ID, software version, timestamp, and scenario configuration hashes |
| `scenario-index.json` | scenario metadata, feasibility, input hashes, and relative sub-run paths |
| `scenario-metrics.parquet` | one status/settlement row per scenario and policy |
| `robustness-metrics.json` | equal-weight feasibility, improvement, mean, and worst-case summaries |
| `pareto-frontier.json` | eligibility rule, objectives, units, and non-dominated policies |
| `report.html` | self-contained human-readable comparison with links to local scenario reports |
| `scenarios/` | one complete, independently validated 12-artifact replay run per scenario |

Publication is atomic. The suite-level `latest` pointer changes only after all summary files and all
scenario sub-runs pass validation. Every scenario also has its own relative `latest` pointer inside
the immutable suite directory.

## Claim boundary

Equal weights are an explicit analysis convention, not estimated occurrence probabilities. Means
over differently scaled workloads or sites can be dominated by larger scenarios even though each
value is baseline-relative. “Worst” means worst among the finite declared scenarios, not a tail-risk
bound. A useful robustness claim therefore requires scenario design, provenance, and coverage that
go beyond merely adding more YAML files.
