# Read-only integration adapters

Phase 3 connects external monitoring and simulation outputs to the same canonical contracts used by
the engineering replay. The adapters only read responses or files. They do not deploy collectors,
change workloads, call control endpoints, or import the upstream platforms as dependencies.

## Prometheus and Kepler

`PrometheusRangeAdapter` accepts the documented Prometheus `query_range` matrix response.
`KeplerPrometheusAdapter` adds explicit queries for the current Kepler `*_watts` gauges documented
in the [Kepler metrics reference](https://github.com/sustainable-computing-io/kepler/blob/main/docs/user/metrics.md).

```python
import pandas as pd

from climadc.adapters import KeplerPrometheusAdapter

result = KeplerPrometheusAdapter().fetch_power(
    base_url="http://prometheus:9090",
    scope="node",
    component="cpu",
    start=pd.Timestamp("2026-08-01T00:00:00Z"),
    end=pd.Timestamp("2026-08-01T01:00:00Z"),
    step=pd.Timedelta(minutes=5),
    site_id="dc-1",
)
telemetry = result.telemetry
```

The generated request uses the read-only [Prometheus HTTP
API](https://prometheus.io/docs/prometheus/latest/querying/api/). Sample time becomes `event_time`;
the completed HTTP retrieval time becomes `available_at`. Kepler output defaults to `estimated`
because the upstream value may combine measurement and model attribution. Set `quality="observed"`
only when the source path provides that evidence.

Supported Kepler scopes are node, container, pod, process, and CPU virtual-machine power; GPU is
supported where Kepler documents a GPU watts gauge. The adapter aggregates zone-level rows with an
explicit `sum by (...)`, preserves the resource identity in `device_id`, rejects negative power,
and does not difference cumulative joule counters. A generic pre-aggregated PromQL expression can
be passed to `PrometheusRangeAdapter` instead.

Credentials must not be embedded in `base_url`, because that URL is retained as lineage metadata.
Use an injected transport to add authentication headers or to parse an exported JSON response.

## Carbon Aware SDK-compatible APIs

`CarbonAwareSDKAdapter` consumes the Web API DTOs documented by the [Green Software Foundation
Carbon Aware SDK](https://github.com/Green-Software-Foundation/carbon-aware-sdk/blob/dev/casdk-docs/docs/tutorial-basics/carbon-aware-webapi.md).
It is provider-neutral: the configured Carbon Aware SDK instance owns the underlying grid provider
and its credentials.

```python
import pandas as pd

from climadc.adapters import CarbonAwareSDKAdapter

adapter = CarbonAwareSDKAdapter()
forecast = adapter.fetch_current_forecast(
    base_url="http://carbon-aware:8080",
    location="eastus",
    site_id="dc-1",
    start=pd.Timestamp("2026-08-01T01:00:00Z"),
    end=pd.Timestamp("2026-08-01T06:00:00Z"),
)
```

The forecast converter preserves `generatedAt` as `issue_time` and uses actual response retrieval
as `available_at`; it rejects a forecast point that was already in the past when the response
arrived. `fetch_observed` reads settlement data separately and defaults to `estimated`, avoiding a
claim that every Carbon Aware provider returns metered observations. Values use the SDK's declared
`gCO2e/kWh` convention.

For archived or externally authenticated responses, use `forecast_from_payload` and
`observed_from_payload`. Those pure converters accept the same DTO shapes without network access.
The adapter does not send the SDK's historical batch POST request.

## SustainDC evaluation exports

`SustainDCAdapter` reads the official `all_agents_episode_*.csv` shape emitted by
[SustainDC](https://github.com/HewlettPackard/dc-rl). It has no Gymnasium, PyTorch, or SustainDC
runtime dependency.

```python
from pathlib import Path

import pandas as pd

from climadc.adapters import SustainDCAdapter

result = SustainDCAdapter().read_evaluation(
    Path("evaluation_data/all_agents_episode_1.csv"),
    site_id="sim-dc",
    region_id="sim-grid",
    start_time=pd.Timestamp("2026-08-01T00:00:00Z"),
    interval=pd.Timedelta(minutes=15),
)
```

| SustainDC field | ClimaDC output | Unit and quality |
|---|---|---|
| `dc_ITE_total_power_kW` | telemetry `it_power` | `kW`, estimated |
| `dc_HVAC_total_power_kW` | telemetry `cooling_power` | `kW`, estimated |
| `dc_total_power_kW` | telemetry `total_power` | `kW`, estimated |
| `outside_temp` | telemetry `air_temperature` | `degC`, estimated |
| `bat_avg_CI` | realized grid `carbon_intensity` | `gCO2e/kWh`, estimated |

The caller supplies the absolute UTC anchor and interval. The adapter checks that upstream
`day`/`hour` rows are unique, ordered, and match that interval. A value becomes available at the end
of its simulation interval. Normalized load-shifting fields are deliberately not converted into a
`FlexibleWorkloadFrame`: the evaluation export does not carry per-job release, deadline, energy,
and power limits.

## Shared boundary

All three integrations return immutable canonical wrappers plus machine-readable metadata. Default
tests use injected responses or local DataFrames and never access the network. These adapters make
external evidence usable in an offline study; they do not turn ClimaDC into a telemetry collector,
simulator, or production scheduler.
