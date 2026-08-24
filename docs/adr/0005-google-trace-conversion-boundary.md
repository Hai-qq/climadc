# ADR 0005: Keep foreign trace conversion separate from E2 claims

- Status: Accepted
- Date: 2026-08-24

## Context

Google ClusterData2019 provides licensed workload events but no task deadlines, task-level measured
power, or climate-signal forecast vintages. Its event time is relative to trace start. Mapping these
events directly into a replay can silently turn future completion facts and scenario assumptions
into apparently causal inputs.

## Decision

ClimaDC supports only offline conversion of a user-exported, bounded v3 CSV. The converter binds the
source hash, exact BigQuery SQL, mapping config, canonical output, and fixed artifact membership. It
accepts unambiguous top-level class 0/1 tasks and fails closed on missing, duplicate, synthesized, or
unsupported events.

Relative submit time becomes release and availability time under a declared scenario epoch. Power,
deadline, and preemptibility remain named scenario mappings. Observed finish time may construct an
ex-post runtime, but the manifest records that it is future information. Every conversion is
`DATA_REQUIRED` and ineligible for a quantitative public claim.

## Consequences

The repository can test and audit conversion behavior without committing multi-terabyte or derived
trace data. Running BigQuery remains an explicit user-controlled, potentially billable acquisition
step. A conversion alone is not E2: forecast vintages, causal availability, independent evaluation
slices, mapping validation, replay verification, and a claim-registry entry remain separate gates.
Google traces cannot be represented as same-site London operational evidence.
