# Engineering input contracts

The v0.2 Alpha semantic spine adds strict inputs for grid signals and flexible batch work. The
contracts are available through `climadc.contracts`; local CSV and Parquet readers are available
through `climadc.adapters`. They are not consumed by the legacy v0.1 shadow scheduler yet.

## Grid signals

`GridSignalFrame` uses exactly these columns:

| Column | Meaning |
|---|---|
| `site_id` | ClimaDC site identifier |
| `region_id` | electricity-grid region identifier, for example `GB-13` |
| `issue_time` | forecast issue time; null for realized rows |
| `available_at` | time the value became available to the study |
| `valid_time` | interval represented by the value |
| `signal` | `carbon_intensity` or `energy_price` |
| `value` | finite numeric value |
| `unit` | emissions/energy or supported currency/energy unit |
| `source` | provider or fixture identifier |
| `quality` | `forecast`, `observed`, or `estimated` |
| `quantile` | optional forecast probability strictly inside `(0, 1)` |

Forecast rows require
`issue_time <= available_at <= valid_time`. Observed and estimated rows require a null
`issue_time`, a null `quantile`, and `valid_time <= available_at`. Carbon intensity must be
nonnegative.

Accepted carbon units are dimensionally compatible with `gCO2e/kWh`, including
`kgCO2e/MWh`. Energy-price units currently support GBP, USD, EUR, and CNY per compatible energy
unit. Currency conversion is deliberately not performed.

Quantile rows remain distinct from point forecasts. If a replay config declares
`risk_quantile=q`, `ClimateForecastFrame` temperature and `GridSignalFrame` price/carbon inputs must
contain an exact `q` row for every required planning slot, causally available at that decision
origin. The replay rejects incomplete or ambiguous quantile scenarios and does not interpolate or
fall back to point forecasts. These rows define an upper-quantile stress scenario, not interval
coverage or a joint probability claim by themselves. After settlement, a risk-enabled study reports
descriptive marginal coverage and loss diagnostics over committed slots; those post-hoc metrics do
not change the input meaning or establish joint calibration.

```python
from pathlib import Path

from climadc.adapters import read_grid_signals

grid = read_grid_signals(
    Path("grid-signals.csv"),
    "csv",
    column_map={},  # the file already uses the canonical columns above
    timezone="UTC",
)
grid_frame = grid.to_pandas()
```

When source columns differ, map each source name to its canonical destination:

```python
grid = read_grid_signals(
    Path("neso-export.csv"),
    "csv",
    column_map={
        "site": "site_id",
        "region": "region_id",
        "issued": "issue_time",
        "retrieved": "available_at",
        "datetime": "valid_time",
        "kind": "signal",
        "forecast": "value",
        "units": "unit",
        "provider": "source",
        "status": "quality",
        "probability": "quantile",
    },
    timezone="Europe/London",
)
```

The mapping must still produce every canonical column exactly once. Provider-specific adapters are
responsible for expanding source records into separate forecast and realized rows. The implemented
Prometheus/Kepler, Carbon Aware SDK-compatible, and SustainDC converters are documented in the
[read-only integration guide](read-only-integrations.md).

## Flexible workload

`FlexibleWorkloadFrame` uses one row per job and exactly these columns:

| Column | Meaning |
|---|---|
| `job_id` | non-empty job identifier, unique within a site |
| `site_id` | ClimaDC site identifier |
| `release_time` | earliest physical execution time |
| `available_at` | time the scheduler learned about the job |
| `deadline` | latest allowed completion time |
| `energy` | positive required IT energy |
| `energy_unit` | unit dimensionally compatible with `kWh` |
| `max_power` | positive per-job IT power limit |
| `power_unit` | unit dimensionally compatible with `kW` |
| `preemptible` | must currently be `true` |
| `priority` | finite, nonnegative scenario priority |

The contract requires `release_time <= available_at <= deadline`. It also checks the continuous
lower bound `energy / max_power` against the execution window from availability to deadline. This
catches jobs that cannot finish even before site-capacity competition is considered.

```python
from climadc.adapters import read_flexible_workload

workload = read_flexible_workload(
    Path("batch-jobs.parquet"),
    "parquet",
    column_map={},
    timezone="UTC",
)
jobs = workload.to_pandas()
```

## Timestamp and ownership rules

- Naive timestamps are interpreted only by local readers using the declared IANA timezone.
- Contract constructors require timezone-aware input and normalize it to exact UTC.
- `to_pandas()` returns a deep copy by default.
- A local reader validates columns, timestamps, values, and units; it does not fabricate missing
  forecast issue times, retrieval times, or job limits.

The [engineering replay kernel](replay-kernel.md) consumes these contracts directly for a
single-site window or rolling sequence. The packaged inputs, source timing boundary, and 12-file
publication contract are documented in the
[Great Britain reference replay](reference-replay.md); see the
[v0.2 design](../design/v0.2-engineering-replay.md) for the overall boundary.
