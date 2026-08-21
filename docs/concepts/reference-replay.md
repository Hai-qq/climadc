# Great Britain reference replay

The packaged `gb-london-carbon-shift-24h` study is an **E1 single-day mechanism demonstration**.
It combines checked-in Open-Meteo/NESO-derived signals with a project workload, declared UTC
tariff, illustrative carbon price, and reference PUE model. It is offline and does not establish
production savings, robustness, marginal avoided emissions, or same-site validation.

```bash
climadc demo carbon-shift --output-dir ./climadc-replay-runs
climadc verify-run ./climadc-replay-runs/latest --json
climadc report ./climadc-replay-runs/latest
python benchmarks/reference/gb_london_24h/reproduce.py --check
```

The compact generated summary and tolerance rationale live under
`benchmarks/reference/gb_london_24h/`. The exact README trade-off statement is bound to claim
`E1-LONDON-TRADEOFF-001` in `evidence/claims.yaml`.

## Inputs and quality

| Input | Decision role | Settlement role | Boundary |
|---|---|---|---|
| Open-Meteo Previous Runs temperature | fixed 24-hour-lead forecast | none | decision availability is a declared scenario assumption |
| Open-Meteo Historical Weather temperature | none | ambient temperature | gridded estimate, not measured site telemetry |
| NESO national GB carbon intensity | forecast | provider estimated actual | national average; two half-hours averaged per UTC hour |
| Declared UTC tariff | forecast price | identical scenario settlement | project scenario, not supplier tariff or bill |
| Four deadline-constrained jobs | workload | completion/SLA accounting | project fixture, not a production trace |

The London label and national GB carbon region are not spatially equivalent. Weather and carbon
settlement remain `estimated`; reports do not shorten them to observed actuals. The emissions field
is `estimated_location_based_emissions_kgco2e`.

## Artifact schema v2

`run-manifest.json` declares the exact directory membership. `environment.json` records runtime and
dependency facts, and `checksums.sha256` covers every other artifact using sorted POSIX-relative
paths. Replay-specific files include portable assumptions, source/lineage records, canonical
Parquet inputs, schedules, profiles, solver status, metrics, and an offline HTML report. Use
`verify-run`, rather than a file-count assertion, to validate the contract and reconstruct the
published numbers.

## Refreshing source bytes

Refresh is an explicit network operation and refuses to overwrite an existing path:

```bash
climadc demo refresh-carbon-shift ./gb-snapshot --decision-date 2026-08-01
climadc replay ./gb-snapshot/study.yaml
```

A new snapshot saves exact bytes as `raw/openmeteo-forecast.json`,
`raw/openmeteo-settlement.json`, and `raw/neso-carbon.json` before parsing.
`raw/retrieval-metadata.json` records public request URLs, HTTP status, allowlisted response headers,
raw hashes, parser/schema versions, transformations, canonical hashes, licenses, and attribution.
Tokens, cookies, authentication headers, and sensitive query fields are never persisted.

The checked-in fixture predates raw capture. Its canonical inputs are hash-bound, but provider bytes
cannot be reconstructed after the fact and are not fabricated. Missing forecast-availability facts
remain scenario assumptions after a fresh historical query.

The packaged four-scenario command is `demo sensitivity-suite`; all scenarios reuse this day and
workload. See [suite semantics](robustness-suites.md).
