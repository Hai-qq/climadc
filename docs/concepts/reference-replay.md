# Great Britain reference replay

The packaged reference study turns the v0.2 contracts and replay kernel into a complete, offline
engineering example. It is a 24-hour, single-origin counterfactual replay for a London reference
site. It does not need an API key or network access.

## Run it

```bash
climadc demo carbon-shift --output-dir ./climadc-replay-runs
climadc report ./climadc-replay-runs/latest
```

The first command prints the immutable run directory. The second prints the path to its static
report without opening a browser. The same inputs can be run through the generic interface:

```bash
climadc replay path/to/study.yaml --output-dir ./replay-runs
```

The package also uses this verified snapshot in a four-scenario objective/demand-charge sensitivity
demo. Run `climadc demo robustness-suite` and see the
[replay robustness suite guide](robustness-suites.md); the extra scenarios are declared assumption
changes, not new independent observations.

## What the fixture contains

| Input | Decision role | Settlement role | Provenance boundary |
|---|---|---|---|
| Open-Meteo [Previous Runs](https://open-meteo.com/en/docs/previous-runs-api) `temperature_2m_previous_day1` | fixed 24-hour-lead weather forecast | none | API data under CC BY 4.0; decision-time availability is a declared scenario assumption |
| Open-Meteo [Historical Weather](https://open-meteo.com/en/docs/historical-weather-api) `temperature_2m` | none | ambient temperature | gridded model/reanalysis estimate, not measured site telemetry |
| [NESO Carbon Intensity API](https://api.carbonintensity.org.uk/) national signal | forecast carbon signal | estimated actual carbon signal | two half-hour values are averaged per hour; original forecast issue/availability times are absent from the historical payload and therefore declared by the scenario |
| Declared time-of-use tariff | price forecast | identical scenario settlement price | project-generated scenario, not a supplier tariff or bill |
| Four deadline-constrained jobs | schedulable workload | completion and SLA accounting | project-generated deterministic fixture, not a production trace |

The site label is London, while the carbon signal is national Great Britain. That spatial mismatch
is explicit in the source manifest and report. The bounded temperature-sensitive PUE model is a
declared reference relationship rather than a calibrated London facility model.

## Auditable output

Each immutable replay run contains exactly 12 files:

| Artifact | Purpose |
|---|---|
| `assumptions.yaml` | portable replay, facility-model, tariff, objective, and limitation assumptions |
| `source-manifest.yaml` | provider URLs, retrieval timestamps, timing bases, licenses, transformations, and SHA-256 hashes bound to the run's Parquet inputs |
| `lineage.json` | run ID, software version, configuration hash, and original verified-input hashes |
| `climate-forecast.parquet` | weather rows legally available to scheduling policies |
| `actual-weather.parquet` | settlement-only estimated weather rows |
| `grid-signals.parquet` | forecast and settlement carbon/price rows with distinct quality and timing |
| `workload.parquet` | normalized jobs and service constraints |
| `schedules.parquet` | per-job allocations for all six policies |
| `profiles.parquet` | forecast and realized PUE, IT power, facility power, price, and carbon by interval |
| `solver-status.json` | policy feasibility and solver status; rolling runs also include decision records and final remaining job energy |
| `replay-metrics.json` | forecast errors, physical/economic/SLA metrics, baseline changes, and Oracle regret |
| `report.html` | self-contained comparison report with inline CSS and no scripts or external assets |

Every number in the HTML comparison is reconstructible from the Parquet and JSON artifacts. Original
input hashes are checked before parsing and retained in lineage; the published source manifest is
then rebound to the run-local Parquet files so the directory remains independently verifiable. A
modified fixture stops the run instead of silently changing the study.

## Refresh a snapshot

Refreshing is optional and explicitly uses the network. It always creates a new directory and
refuses to overwrite an existing path:

```bash
climadc demo refresh-carbon-shift ./gb-snapshot --decision-date 2026-08-01
climadc replay ./gb-snapshot/study.yaml
```

The refresh command queries the Open-Meteo Previous Runs and Archive APIs plus the NESO Carbon
Intensity API, normalizes the responses into the same contracts, and writes a new hash-bound source
manifest. It does not improve the missing historical issue/availability evidence: those fields
remain visibly classified as scenario assumptions.

## Interpretation

The report may show a policy reducing one outcome while increasing another. Signed changes are not
filtered, and the joint objective is a weighted comparison score rather than money. Because the
tariff, workload, and facility model are scenarios, the result demonstrates causal replay and
engineering accounting—not measured production savings.
