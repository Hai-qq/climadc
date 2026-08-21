# Quickstart

## Prerequisites

- Python 3.10, 3.11, 3.12, or 3.13.
- Bash or Zsh for the tested six-line block. Windows users can use the PowerShell equivalent below.
- An installed checkout: `python -m pip install -e .`. After an authorized Alpha release, the equivalent version is `climadc==0.3.0a1`.

The built-in study is deterministic, project-owned synthetic data. It needs no API key or network access.

## Tested and independently verified path

Run this block from any directory after installation. The marker on the fence is consumed by the offline documentation test.

```bash quickstart-test
export CLIMADC_STUDY="$(mktemp -d)/climadc-quickstart"
climadc init "$CLIMADC_STUDY"
climadc validate "$CLIMADC_STUDY/study.yaml"
climadc benchmark "$CLIMADC_STUDY/study.yaml"
climadc verify-run "$CLIMADC_STUDY/runs/latest"
climadc report "$CLIMADC_STUDY/runs/latest"
```

The final command prints the absolute path to `report.html`.

## PowerShell equivalent

```powershell
$Study = Join-Path ([System.IO.Path]::GetTempPath()) ("climadc-" + [guid]::NewGuid())
climadc init $Study
climadc validate (Join-Path $Study "study.yaml")
climadc benchmark (Join-Path $Study "study.yaml")
climadc verify-run (Join-Path $Study "runs/latest")
climadc report (Join-Path $Study "runs/latest")
```

## Published run

`runs/latest` resolves to an immutable schema v2 run directory. Its manifest declares the complete
file set; the current benchmark payload is:

| Artifact | Purpose |
|---|---|
| `run.yaml` | frozen configuration and run identity |
| `lineage.json` | input hashes, model IDs, split IDs, and software version |
| `splits.parquet` | explicit train/calibration/test membership |
| `predictions.parquet` | typed point and quantile predictions |
| `metrics.json` | forecast and shadow-decision metrics |
| `leakage-report.json` | accepted/rejected rows and violations |
| `dataset-card.md` | source, license, hash, range, and limitations |
| `report.html` | static human-readable report |
| `run-manifest.json` | versioned run identity, config/input hashes, Git state, and declared files |
| `environment.json` | Python, platform, dependency, timezone, seed, and constraints facts |
| `checksums.sha256` | sorted relative SHA-256 coverage of every other artifact |

Use `verify-run`, rather than a fixed file-count assertion, to validate this contract. The synthetic
Quickstart checks execution, contracts, and artifact integrity. It does not demonstrate real-site
accuracy or savings.

## Engineering replay demo

The v0.3 development checkout also ships a hash-verified Great Britain reference replay. It uses a
checked-in Open-Meteo/NESO-derived snapshot plus a declared tariff, workload, and PUE model, so the
command is fully offline:

```bash
climadc demo carbon-shift --output-dir ./climadc-replay-runs
climadc verify-run ./climadc-replay-runs/latest
climadc report ./climadc-replay-runs/latest
```

This path publishes a versioned schema v2 replay record. See the
[Great Britain reference replay guide](concepts/reference-replay.md) for the artifact contract, source
timing assumptions, optional network refresh, and claim boundary.

## Rolling and upper-quantile replay

The generic replay command also supports the Phase 4 options below in `study.yaml`:

```yaml
replay:
  # existing replay fields...
  risk_quantile: 0.9
rolling:
  periods: 24
  step: 1h
```

Run it through the same CLI:

```bash
climadc replay ./study.yaml --output-dir ./replay-runs
climadc report ./replay-runs/latest
```

The packaged Great Britain fixture remains a six-policy, single-window example because it does not
declare quantile inputs. A risk-enabled study must provide exact, causally available temperature,
price, and carbon rows at the configured quantile for every planning slot. See the
[engineering replay kernel](concepts/replay-kernel.md) for rolling state semantics and current
deadline limits.

When `risk_quantile` is enabled, the same command also writes committed-slot marginal diagnostics
to `replay-metrics.json` and `report.html`: empirical coverage, a 95% Wilson interval, coverage gap,
exceedance magnitudes, and pinball loss. No extra configuration is required, and the diagnostics do
not turn the three marginal scenarios into a joint probability guarantee.

## Multi-scenario sensitivity suite

Run the packaged four-scenario policy-sensitivity study fully offline:

```bash
climadc demo sensitivity-suite --output-dir ./climadc-replay-suite-runs
climadc verify-suite ./climadc-replay-suite-runs/latest
climadc report ./climadc-replay-suite-runs/latest
```

For a custom matrix, create a `suite.yaml` whose scenarios point to complete replay study YAML
files, then run `climadc replay-suite ./suite.yaml`. The suite publishes a v2 manifest, checksums,
machine-readable scenario rows, equal-weight sensitivity summaries, a Pareto frontier, a
self-contained report, and one independently verifiable v2 sub-run per scenario. See the
[replay suite guide](concepts/robustness-suites.md) for compatibility gates and claim
limits.

The old `demo robustness-suite` spelling remains a deprecated alias. Only matrices that actually
vary a declared date, season, location, or workload dimension may use `suite_type: robustness`.
