# Quickstart

## Prerequisites

- Python 3.10, 3.11, 3.12, or 3.13.
- Bash or Zsh for the tested five-command block. Windows users can use the PowerShell equivalent below.
- An installed checkout: `python -m pip install -e .`. After the Alpha is published, `python -m pip install "climadc==0.2.0a1"` is equivalent.

The built-in study is deterministic, project-owned synthetic data. It needs no API key or network access.

## Tested five-command path

Run this block from any directory after installation. The marker on the fence is consumed by the offline documentation test.

```bash quickstart-test
export CLIMADC_STUDY="$(mktemp -d)/climadc-quickstart"
climadc init "$CLIMADC_STUDY"
climadc validate "$CLIMADC_STUDY/study.yaml"
climadc benchmark "$CLIMADC_STUDY/study.yaml"
climadc report "$CLIMADC_STUDY/runs/latest"
```

The final command prints the absolute path to `report.html`.

## PowerShell equivalent

```powershell
$Study = Join-Path ([System.IO.Path]::GetTempPath()) ("climadc-" + [guid]::NewGuid())
climadc init $Study
climadc validate (Join-Path $Study "study.yaml")
climadc benchmark (Join-Path $Study "study.yaml")
climadc report (Join-Path $Study "runs/latest")
```

## Published run

`runs/latest` resolves to an immutable run directory containing exactly:

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

The synthetic Quickstart checks execution, contracts, and artifact integrity. It does not demonstrate real-site accuracy or savings.

## Engineering replay demo

The v0.2 development checkout also ships a hash-verified Great Britain reference replay. It uses a
checked-in Open-Meteo/NESO-derived snapshot plus a declared tariff, workload, and PUE model, so the
command is fully offline:

```bash
climadc demo carbon-shift --output-dir ./climadc-replay-runs
climadc report ./climadc-replay-runs/latest
```

This path publishes a separate 12-artifact replay record. See the
[Great Britain reference replay guide](concepts/reference-replay.md) for the exact files, source
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

## Multi-scenario robustness suite

Run the packaged four-scenario policy-sensitivity study fully offline:

```bash
climadc demo robustness-suite --output-dir ./climadc-replay-suite-runs
climadc report ./climadc-replay-suite-runs/latest
```

For a custom matrix, create a `suite.yaml` whose scenarios point to complete replay study YAML
files, then run `climadc replay-suite ./suite.yaml`. The suite publishes eight top-level entries,
including machine-readable scenario rows, equal-weight robustness summaries, a Pareto frontier, a
self-contained report, and one full 12-artifact sub-run per scenario. See the
[replay robustness suite guide](concepts/robustness-suites.md) for compatibility gates and claim
limits.
