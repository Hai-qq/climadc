# WeatherDC Kasetsart reference study

This reference study demonstrates how a climate/DC project can enter ClimaDC without importing
its generated models or processed outputs. It converts the four WeatherDC meter streams and five
HII weather variables into canonical contracts, then runs an auditable forecast-to-decision study.

## Offline small mode

From a fresh checkout, install the package first and then run the example from the repository root:

```bash
python -m pip install -e .
```

```bash
python examples/weatherdc_kasetsart/run.py --small
```

Small mode never accesses the network. It uses the deterministic, project-owned fixture in
`tests/fixtures/weatherdc_small`, whose `PROVENANCE.yaml` declares `CC0-1.0` and confirms that no
operational records were copied. The command prints the published run directory. That directory
contains `run.yaml`, lineage, temporal splits, predictions, metrics, leakage audit, dataset cards,
and HTML/Markdown reports.

The reported `weatherdc` model is a deliberately small external-model example, not a renamed
ClimaDC baseline. It fits ordinary least squares on temperature, humidity, and solar forecast
features that were already available at the common decision origin. It is compared with a legal
last-observation persistence forecast on the same test timestamps. Both models' point predictions
are written to `predictions.parquet`; `metrics.json` is computed from those exact values.

The synthetic forecast batch has an explicit issue/availability time before the study window. This
is why the small reference can require zero rejected climate rows. This availability convention is
specific to the synthetic fixture and is not backfilled onto HII observations.

`benchmarks/weatherdc.yaml` is exclusively the small-mode benchmark template. Its generated input
paths are replaced by `run.py`; it is not a configuration for the upstream conversion.

## Verified upstream conversion only

```bash
python examples/weatherdc_kasetsart/run.py --full
```

`--full` is explicitly conversion-only. It downloads the ten immutable sources listed in
`data-card.md` to
`.cache/climadc/weatherdc/raw`. Every file is checked against both its byte count and SHA-256 before
an atomic rename. Existing cache files are reused only after the same checks. The command then
writes canonical climate and telemetry CSVs under `.cache/climadc/weatherdc/study`. It does not
create or infer workload data and does not run a benchmark. The generated
`conversion-manifest.yaml` records the measured download-plus-conversion runtime, exact source
URLs/hashes, output hashes, observation timing semantics, and the absence of benchmark/workload
outputs.

Full mode is intentionally a download/conversion entrypoint, not a claim that observed HII weather
is a forecast. Because the HII CSVs contain observations without issue or retrieval timestamps,
the adapter records `issue_time = available_at = valid_time`. A real full reference forecast must
first produce causally timestamped weather forecasts; the original WeatherDC weather-model
retraining is outside this Alpha's offline quickstart and its runtime is not represented by the
conversion timing.

Review the upstream license and redistribution terms before using the downloaded files. ClimaDC
does not redistribute or relicense them.

## Limits

- The original WeatherDC weather station is about 30.5 km from the data-center location.
- Raw meter files do not expose ingestion timestamps. The converter conservatively records hourly
  meter observations as available at their event time and documents that assumption.
- Invalid timestamps, non-numeric readings, the HII `-999` missing sentinel, invalid humidity, and
  negative rain/solar readings are excluded; targets are never interpolated.
- The small fixture proves framework behavior only. Its scores are not evidence of operational
  accuracy, energy savings, or performance on the upstream dataset.
