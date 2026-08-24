# Google ClusterData2019 conversion assets

This directory contains only a bounded BigQuery export template and a conversion-config template.
It does **not** contain Google trace rows or a completed E2 benchmark.

- `export_workload.sql` summarizes one cell's task submit/finish events into the exact CSV schema
  accepted by `climadc trace convert-google-v3`.
- `conversion.example.yaml` declares the source hash, relative-time mapping, supported scheduling
  classes, and explicit power/deadline/preemptibility assumptions.

Review BigQuery's dry-run byte estimate before executing the query. The public 2019 trace is large,
requires a billing-enabled Google Cloud project, and is licensed CC BY 4.0. Keep raw/exported data
outside Git. The complete procedure and evidence limitations are documented in
[`docs/concepts/trace-driven-benchmarks.md`](../../docs/concepts/trace-driven-benchmarks.md).

Official sources:

- [Google ClusterData2019 trace documentation](https://github.com/google/cluster-data/blob/master/ClusterData2019.md)
- [v3 trace-format proto](https://github.com/google/cluster-data/blob/master/clusterdata_trace_format_v3.proto)
- [official BigQuery analysis notebook](https://github.com/google/cluster-data/blob/master/clusterdata_analysis_colab.ipynb)
- [Google ClusterData2019 power trace](https://github.com/google/cluster-data/blob/master/PowerData2019.md)
