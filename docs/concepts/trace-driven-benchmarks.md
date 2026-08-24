# Trace-driven benchmark preparation

ClimaDC can convert a **user-exported, bounded** Google ClusterData2019 v3 task slice into a
`FlexibleWorkloadFrame`. This is offline conversion infrastructure, not a bundled dataset and not
an E2 result. Every conversion manifest is deliberately marked `DATA_REQUIRED` and
`claim_eligible: false`.

## Source and acquisition boundary

Google describes the 2019 release as traces from eight Borg cells collected in May 2019 and
publishes it through BigQuery under CC BY 4.0. The compressed trace is approximately 2.4 TiB, so a
query can incur cost and requires a billing-enabled Google Cloud project. ClimaDC never runs that
query automatically, handles credentials, or places exported rows in Git.

Use the official [trace documentation](https://github.com/google/cluster-data/blob/master/ClusterData2019.md),
[v3 proto](https://github.com/google/cluster-data/blob/master/clusterdata_trace_format_v3.proto),
and [BigQuery notebook](https://github.com/google/cluster-data/blob/master/clusterdata_analysis_colab.ipynb)
as the schema authorities. Google's separate
[2019 power trace](https://github.com/google/cluster-data/blob/master/PowerData2019.md) covers 57
power domains, most corresponding to the eight trace cells, but it does not supply a task-level
power attribution for this converter.

## Bounded export

Start from [`export_workload.sql`](https://github.com/Hai-qq/climadc/blob/main/benchmarks/google_clusterdata_2019/export_workload.sql).
It accepts three named `INT64` parameters:

| Parameter | Meaning |
|---|---|
| `start_time_us` | Inclusive submit-window start, relative to trace start |
| `end_time_us` | Exclusive submit-window end |
| `finish_cutoff_time_us` | Exclusive event scan cutoff; must be at least `end_time_us` |

First use BigQuery dry-run mode and inspect the estimated bytes. Execute only after accepting the
cost and data terms. Cell `a` is selected in the template; changing it requires the same cell value
in the conversion config. The exact executed SQL must be retained, not regenerated after export.

Export CSV with exactly these columns:

```text
collection_id,instance_index,submit_time_us,finish_time_us,requested_cpu,priority,scheduling_class,missing_type,collection_type,alloc_collection_id,submit_count,finish_count
```

The converter fails closed on empty or extra columns, duplicate task keys, absent or multiple
submit/finish events, synthesized missing events, unsupported latency classes, alloc-set tasks,
non-positive CPU requests, and events outside the declared window. Only top-level job tasks in
scheduling classes 0 and/or 1 are accepted. Class 0 is best effort and class 1 is commonly used for
batch work in the upstream proto; classes 2 and 3 are latency-sensitive and are excluded.

## Conversion and independent verification

Copy [`conversion.example.yaml`](https://github.com/Hai-qq/climadc/blob/main/benchmarks/google_clusterdata_2019/conversion.example.yaml)
outside the repository. Replace its zero hash with the SHA-256 of the exact CSV bytes, record the
actual UTC export time, review every mapping, and retain a private or external immutable copy of
the source CSV.

```bash
climadc trace convert-google-v3 google-v3-a.csv conversion.yaml google-v3-a-conversion \
  --query-sql executed-export.sql
climadc trace verify-google-v3 google-v3-a-conversion --source-csv google-v3-a.csv
```

The conversion directory contains `conversion-config.yaml`, `conversion-manifest.json`,
`export-query.sql`, `workload.csv`, and `checksums.sha256`. Publication is atomic and refuses to
overwrite an existing path. Verification without `--source-csv` validates membership, checksums,
config/query/manifest agreement, and the canonical workload contract. Supplying the source also
repeats the conversion byte-for-byte.

## Mapping semantics and limitations

| Canonical field | v3 input or assumption | Evidence boundary |
|---|---|---|
| `release_time`, `available_at` | submit time plus declared UTC scenario epoch | Epoch is a scenario mapping, not a Google wall-clock timestamp |
| `energy` | requested normalized CPU × declared kW/CPU × observed runtime × utilization fraction | Scenario estimate; not measured task or facility energy |
| `max_power` | requested normalized CPU × declared kW/CPU | Scenario estimate; normalized CPU is not a physical core count |
| `deadline` | submit time + observed runtime × declared multiplier | Uses a future finish event during scenario construction |
| `preemptible` | explicit `true` assumption | Not asserted by the trace |
| `priority` | submit-event priority | Preserved source field |

Observed completion is future information relative to submission. The converter records that fact
in the manifest and uses it only to construct an ex-post scenario. It must not be presented as a
deadline or energy value known by the original scheduler. A causal E2 benchmark must either justify
this as a predeclared scenario abstraction and sensitivity-test it, or replace it with historically
available deadline/runtime information.

## What still blocks E2

A verified conversion bundle satisfies only part of the workload-provenance gate. E2 still needs:

- licensed immutable source retention and attribution;
- prospective or historically archived forecast vintages with real availability timestamps;
- a predeclared relationship between the foreign workload timeline and climate/grid scenario;
- independent train/calibration/evaluation dates, sites, or workload slices;
- defensible power, deadline/slack, and preemptibility mappings with sensitivity analysis; and
- a complete replay artifact plus a hash-bound claim registry entry.

Google workload and power traces must not be described as same-site London history. Until all gates
pass, no conversion or downstream replay may be labelled an E2 result.
