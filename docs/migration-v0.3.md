# Migrating to v0.3 Alpha

## Verify published directories

New benchmark and replay writers emit artifact schema v2. Replace assertions about “exactly 8” or
“exactly 12” files with `climadc verify-run RUN --json`; use `verify-suite` for suite roots. V1
directories remain read-only and return `legacy: true` with explicit limitations.

## Replace unscaled weights

Old configuration is accepted unchanged and emits a deprecation warning:

```yaml
replay:
  cost_weight: 1
  carbon_weight: 0.1
  demand_charge_per_kw: 0
```

Because this adds currency and kgCO2e without a conversion, migrate by declaring the intended
exchange rate. The old example above is schedule-equivalent to a 100 currency/tCO2e carbon price:

```yaml
replay:
  objective:
    version: "1"
    mode: monetized
    carbon_price_currency_per_tco2e: 100
    demand_charge_per_kw: 0
```

Use `epsilon_constraint` to minimize cost under explicit emissions/peak bounds, or
`pareto_analysis` with a sorted, unique fixed list of carbon prices. Pareto analysis is currently
single-window only; rolling configs fail explicitly rather than silently changing semantics.

## Account for deterministic job-allocation tie-breaking

The v0.3 solver keeps every policy's primary result and aggregate slot power unchanged, then uses a
fixed ASAP ordering to choose among equivalent job-level allocations. A replay produced by an older
build can therefore have a different schedule row assignment or `shifted_energy_kwh` even when its
energy, cost, emissions, and peak metrics are identical. Regenerate compact reference artifacts when
byte-for-byte comparison is required.

## Rename fields and suites

- `emissions_kgco2e` becomes `estimated_location_based_emissions_kgco2e` in v2 replay outputs.
- Corresponding ASAP delta and suite aggregate names use the same prefix.
- `suite_type` is required by intent. The packaged suite is `sensitivity`.
- Use `demo sensitivity-suite`; `demo robustness-suite` is a deprecated alias.
- `robustness-metrics.json` becomes `suite-metrics.json` for the generalized suite contract.
- `ReplaySuiteResult.suite_metrics` is the canonical API; the old `robustness_metrics` accessor
  remains as a deprecated read-only alias.

The fixture values are not silently changed: its tariff is now explicitly documented as a declared
UTC tariff scenario.
