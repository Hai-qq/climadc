# ADR 0004: Separate evidence levels and sensitivity from robustness

- Status: Accepted
- Date: 2026-08-21

## Decision

Project synthetic fixtures are E0, external-signal/synthetic-workload replay is E1, trace-driven
causal benchmarks are E2, and same-site operational validation is E3. A suite that reuses one day
and workload while changing objectives is sensitivity analysis. Robustness requires independent
sample dimensions and provenance.

## Consequences

Current artifacts cannot be labelled E2/E3. The old robustness demo name remains only as a warned
compatibility alias. Quantitative README statements link to the machine-readable claim registry.
