# Time semantics

ClimaDC keeps three different questions explicit:

- `issue_time`: when a forecast was produced by its model or provider.
- `available_at`: when that exact row could actually be read by the downstream experiment.
- `valid_time`: when the predicted physical condition occurs.

For a climate forecast, the legal ordering is:

```text
issue_time <= available_at <= valid_time
```

At a decision origin `T`, a feature row is usable only when `available_at <= T`. Comparing only `valid_time` or `issue_time` is insufficient.

## Legal forecast example

Suppose a provider issues a temperature forecast at 08:00, the file reaches the experiment at 08:07, it predicts 12:00, and the scheduling decision runs at 09:00.

| field | UTC value |
|---|---|
| `issue_time` | 2026-07-11 08:00 |
| `available_at` | 2026-07-11 08:07 |
| decision origin | 2026-07-11 09:00 |
| `valid_time` | 2026-07-11 12:00 |

This row is legal: it was issued and delivered before the decision, and it predicts a future valid time.

## Illegal delayed-delivery example

The same 12:00 forecast is not legal for the 09:00 decision if it was backfilled at 09:20:

| field | UTC value |
|---|---|
| `issue_time` | 2026-07-11 08:00 |
| decision origin | 2026-07-11 09:00 |
| `available_at` | 2026-07-11 09:20 |
| `valid_time` | 2026-07-11 12:00 |

Its old `issue_time` does not make it available earlier. `LeakageGuard` rejects it at the 09:00 origin.

## Illegal observation-as-forecast example

An observed temperature at 12:00 can honestly use:

```text
issue_time = available_at = valid_time = 2026-07-11 12:00 UTC
```

It cannot be used as a 09:00 forecast by changing `issue_time` or `available_at` to 09:00. That would backdate knowledge. This is why the WeatherDC full path is conversion-only: its upstream HII rows are observations without historical issue/retrieval timestamps.

## Telemetry and labels

Telemetry uses `event_time` rather than `valid_time`. A measurement must satisfy `event_time <= available_at`. Training labels are causal only when `available_at` is no later than the training origin; post-hoc test evaluation may read the eventual observed target, but model fitting may not.

All internal timestamps are timezone-aware UTC. Naive timestamps are rejected unless an adapter has an explicit source timezone with which to localize them.
