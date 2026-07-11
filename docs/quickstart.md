# Quickstart

## Prerequisites

- Python 3.10, 3.11, 3.12, or 3.13.
- Bash or Zsh for the tested five-command block. Windows users can use the PowerShell equivalent below.
- An installed checkout: `python -m pip install -e .`. After the Alpha is published, `python -m pip install "climadc==0.1.0a1"` is equivalent.

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
