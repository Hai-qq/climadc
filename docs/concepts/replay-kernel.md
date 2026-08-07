# Engineering replay kernel

The unreleased v0.2 replay kernel turns the canonical engineering inputs into a constrained,
counterfactual comparison. It can solve one single-site decision window or repeatedly re-solve a
rolling horizon, then settles every committed schedule against realized weather, energy price, and
carbon intensity. It never sends a schedule to production infrastructure.

## What is implemented

`ReplayEngine` compares six policies by default under identical job and IT-capacity constraints.
Setting `risk_quantile` adds a seventh:

| Policy | Decision-time objective |
|---|---|
| `asap` | execute accepted work in the earliest eligible intervals, with higher priority first under contention; this is the baseline |
| `peak` | minimize forecast facility peak |
| `price` | minimize forecast energy charge plus the configured demand charge |
| `carbon` | minimize forecast operational emissions |
| `joint` | minimize the configured weighted cost-and-carbon objective |
| `risk_aware` | minimize the same joint objective using the declared upper-quantile temperature, price, and carbon scenario |
| `oracle` | minimize the same joint objective using realized inputs; hindsight comparator only |

All jobs known by the exact UTC decision time are hard constraints: each accepted job must conserve
its required IT energy while respecting release time, deadline, maximum power, and shared IT
capacity. Jobs whose `available_at` is later than the decision time are excluded and counted as
`future_jobs`. An arrived job is never silently dropped.

Priority orders contended ASAP work; it is not a license to drop lower-priority jobs. Every accepted
job remains a hard completion constraint for every policy.

The reference `TemperatureSensitivePUEModel` uses a bounded linear curve:

```text
pue[t] = clip(
    base_pue + slope_per_degree_c * (temperature_c[t] - reference_temperature_c),
    min_pue,
    max_pue,
)
facility_power_kw[t] = pue[t] * (fixed_it_power_kw + flexible_it_power_kw[t])
```

`FacilityEnergyModel` is a public protocol, so a calibrated site model can replace the reference
curve without changing the optimizer.

## Minimal API flow

Construct the four strict contracts described in the
[engineering input guide](engineering-inputs.md), then run one decision window:

```python
import pandas as pd

from climadc.replay import ReplayConfig, ReplayEngine, TemperatureSensitivePUEModel

config = ReplayConfig(
    site_id="dc-1",
    horizon=pd.Timedelta(hours=24),
    interval=pd.Timedelta(hours=1),
    it_capacity_kw=500.0,
    fixed_it_power_kw=300.0,
    cost_weight=1.0,
    carbon_weight=1.0,
    demand_charge_per_kw=0.0,
)

result = ReplayEngine(TemperatureSensitivePUEModel()).run(
    decision_time=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
    climate_forecast=climate,
    actual_weather=weather,
    grid_signals=grid,
    workload=workload,
    config=config,
)

print(result.status)
print(result.metrics)
```

The decision time must already be an exact UTC `pandas.Timestamp`. The horizon must be an integer
multiple of the interval. Input timestamps are interval starts; a job can use a slot only when its
start is not earlier than the job release and its end is not later than the deadline.

The study runner exposes the same engine through `climadc replay`. Add these optional fields to a
study YAML to enable both Phase 4 features:

```yaml
replay:
  # existing replay fields...
  risk_quantile: 0.9
rolling:
  periods: 24
  step: 1h
```

`rolling.periods` is the number of decision origins. `rolling.step` must be a positive integer
multiple of the replay interval and cannot exceed the full horizon. Omitting `rolling` preserves
single-window behavior; omitting `risk_quantile` preserves the original six-policy comparison.

## Causal selection and units

For each required slot, the engine selects the latest point forecast whose `issue_time` and
`available_at` are not later than the decision time. A null quantile is preferred; a median
(`quantile=0.5`) is accepted when no null-quantile point exists. Equally fresh alternatives are
rejected as ambiguous instead of being selected by row order.

When `risk_quantile=q` is configured, the risk-aware policy independently selects the latest exact
`q` forecast for temperature, energy price, and carbon intensity at every slot. The value must be
strictly inside `(0.5, 1)`. Missing, ambiguous, or differently labelled quantiles are errors; the
engine neither interpolates quantiles nor falls back to point forecasts. Combining three marginal
upper quantiles produces one declared stress scenario. It is not a joint probability guarantee,
CVaR, or calibrated two-sided prediction interval.

Realized rows are selected separately and are used only for settlement and the Oracle comparator.
The kernel normalizes compatible units to:

- temperature: `degC`;
- IT energy and power: `kWh` and `kW`;
- carbon intensity: `kgCO2e/kWh`;
- energy price: one consistent supported currency per `kWh`.

No currency conversion is attempted. Mixing currencies in one replay is an error.
`demand_charge_per_kw` is interpreted in that same currency and applied to this decision window's
realized peak. It is a scenario assumption, not a reconstruction of a utility's monthly tariff.

## Post-hoc upper-quantile diagnostics

When `risk_quantile=q` is configured and settlement succeeds, the study runner backtests the
temperature, energy-price, and carbon-intensity quantiles separately. A single-window run uses its
settled slots; a rolling run uses only committed slots and the quantile view selected at each
origin. Planned but uncommitted horizon rows never enter the sample.

For each marginal signal, `replay-metrics.json` and `report.html` publish:

- inclusive empirical coverage, counting `actual <= quantile_forecast` as covered;
- the signed gap `empirical_coverage - q`, so a negative value means undercoverage;
- a two-sided 95% Wilson binomial interval for the empirical coverage proportion;
- exceedance count, mean positive exceedance across all slots, conditional mean exceedance, and
  maximum exceedance in the canonical signal unit;
- mean pinball loss at `q` in that same unit.

The sample count is explicit because a short replay can produce a wide uncertainty interval. These
are post-hoc marginal diagnostics: they do not alter decisions, fit or recalibrate quantiles, supply
a lower bound, establish joint `q` coverage across the three signals, or make the stress scenario a
CVaR objective. Two-sided interval coverage remains unavailable because the replay accepts only one
declared upper quantile for this policy.

## Result tables

`ReplayResult` and `RollingReplayResult` are frozen, and every DataFrame property returns a
defensive copy:

- `status`: solver feasibility and message for every policy;
- `allocations`: job-by-slot IT power and energy;
- `profiles`: forecast/actual temperature, PUE, grid signals, IT power, and facility power;
- `metrics`: realized `kWh`, `kgCO2e`, cost, peak `kW`, SLA, shifted energy, baseline changes, and
  Oracle regret;
- `violations`: explicit reasons when the decision window is infeasible.

At the study level, `forecast_metrics` also contains point-forecast MAE and, when configured, the
upper-quantile diagnostics above. The same payload is nested under `forecast` in
`replay-metrics.json`.

The rolling result also exposes `decisions`, including policy-level solver status and committed
energy at each origin, plus `remaining_energy` for the final per-policy job state. Its schedules and
profiles contain `decision_time`, so every committed row can be traced to the forecast view that
created it. The existing 12-artifact publication contract records rolling mode, decision count,
commit interval, and decision-level solver records.

Settlement uses realized facility energy:

```text
facility_energy_kwh = sum_t(actual_pue[t] * total_it_power_kw[t] * interval_hours)
emissions_kgco2e = sum_t(facility_energy_kwh[t] * actual_carbon_kgco2e_per_kwh[t])
energy_charge = sum_t(facility_energy_kwh[t] * actual_energy_price[t])
demand_charge = actual_peak_kw * demand_charge_per_kw
```

`objective_regret` is the policy's realized weighted objective minus the Oracle objective. It is
nonnegative for a single window, where every policy sees the same accepted jobs. In rolling mode,
the Oracle knows realized signals at each origin but still obeys job `available_at`; a later unknown
arrival can make another policy's cumulative signed difference negative. Interpret the rolling
field as an Oracle delta, not a global perfect-foresight regret bound. The unweighted cost,
emissions, energy, and peak values remain available even when a joint objective is used. A
cost-plus-carbon objective is a comparison score—not a currency amount—unless `carbon_weight` is
zero; its weights encode the user's chosen trade-off between unlike quantities.
`shifted_energy_kwh` is half the job-by-slot L1 distance from the ASAP allocation, so moving one kWh
from one slot to another counts as one shifted kWh rather than two.

## Rolling behavior and current boundary

`RollingReplayEngine` is receding-horizon orchestration: at each decision origin it selects only
inputs causally available at that origin, re-solves the complete configured horizon for every
policy, commits only `rolling.step`, and carries remaining job energy separately for each policy.
Only committed slots enter aggregate energy, cost, emissions, peak, SLA, shifted-energy, and Oracle
regret metrics. This prevents later re-optimization from double-counting planned but uncommitted
energy.

The Phase 2 study runner, historical Open-Meteo/NESO adapters, CLI, hash-bound replay artifacts, and
comparative HTML report all support this mode; see the
[Great Britain reference replay](reference-replay.md) for the packaged single-window example.
Current limits are explicit:

- execution remains single-site, offline, and non-actuating;
- once a job is accepted at an origin, its deadline must fit inside that origin's complete planning
  horizon; longer-lived or terminal carry-over jobs are rejected rather than truncated;
- the upper-quantile policy remains a user-declared stress scenario, not a fitted
  distributional-risk model or a joint coverage guarantee; the emitted marginal diagnostics are
  descriptive only;
- each rolling solve optimizes its current horizon's demand-charge term, while final settlement
  applies the charge once to the peak across committed slots; this is not dynamic programming over
  a reconstructed monthly billing maximum.

Realized values are counterfactual settlement inputs, not evidence that a production site achieved
the same savings.

To compare these replay policies across several complete studies without losing scenario-level
provenance, use the [replay robustness suite](robustness-suites.md). The suite is orchestration and
aggregation over this kernel; it does not change solver semantics.
