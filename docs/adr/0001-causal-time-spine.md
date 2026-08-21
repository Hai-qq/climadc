# ADR 0001: Preserve a three-timestamp causal spine

- Status: Accepted
- Date: 2026-08-21

## Decision

Forecast and grid decisions retain `issue_time`, `available_at`, and `valid_time` as separate facts.
Telemetry uses event/valid time plus availability. A candidate decision can consume only rows whose
availability is no later than its origin. Estimated settlement is not relabelled as observation.

## Consequences

Historical data without issue/retrieval facts cannot be backdated into a forecast. WeatherDC full
mode remains conversion-only. This restriction is central to both LeakageGuard and replay evidence.
