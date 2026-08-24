# 轨迹驱动 Benchmark 准备

ClimaDC 可以把用户自行导出的、**有界的** Google ClusterData2019 v3 任务切片转换为
`FlexibleWorkloadFrame`。这只是离线转换基础设施，不是仓库内置数据集，也不是 E2 结果。
每个转换清单都会明确写入 `DATA_REQUIRED` 与 `claim_eligible: false`。

## 来源与获取边界

Google 将 2019 数据描述为 2019 年 5 月八个 Borg cell 的轨迹，通过 BigQuery 提供并采用
CC BY 4.0。压缩数据约 2.4 TiB，因此查询可能产生费用，且需要启用结算的 Google Cloud
项目。ClimaDC 不会自动执行查询、处理凭据或把导出行写入 Git。

模式权威来源是官方[轨迹文档](https://github.com/google/cluster-data/blob/master/ClusterData2019.md)、
[v3 proto](https://github.com/google/cluster-data/blob/master/clusterdata_trace_format_v3.proto) 与
[BigQuery notebook](https://github.com/google/cluster-data/blob/master/clusterdata_analysis_colab.ipynb)。
Google 另行发布的 [2019 功率轨迹](https://github.com/google/cluster-data/blob/master/PowerData2019.md)
包含 57 个 power domain，其中大多数对应八个轨迹 cell；但它没有为本转换器提供任务级功率归因。

## 有界导出

从 [`export_workload.sql`](https://github.com/Hai-qq/climadc/blob/main/benchmarks/google_clusterdata_2019/export_workload.sql) 开始。
它接收三个命名 `INT64` 参数：

| 参数 | 含义 |
|---|---|
| `start_time_us` | 相对轨迹起点的提交窗口（含） |
| `end_time_us` | 提交窗口终点（不含） |
| `finish_cutoff_time_us` | 事件扫描终点（不含），不得早于 `end_time_us` |

先执行 BigQuery dry run 并检查预估扫描字节；只有接受费用与数据条款后才执行。模板选择 cell
`a`，若修改 cell，转换配置必须同步修改。应保存真正执行的 SQL，不能在导出后重新生成。

CSV 必须且只能包含以下列：

```text
collection_id,instance_index,submit_time_us,finish_time_us,requested_cpu,priority,scheduling_class,missing_type,collection_type,alloc_collection_id,submit_count,finish_count
```

转换器会拒绝空列或额外列、重复任务键、缺失或多个 submit/finish、合成补齐事件、不支持的
延迟等级、alloc-set 任务、非正 CPU 请求以及窗口外事件。只接受顶层 job task 与调度等级
0/1；上游 proto 中 0 是 best effort，1 常用于 batch，延迟敏感的 2/3 被排除。

## 转换与独立验证

把 [`conversion.example.yaml`](https://github.com/Hai-qq/climadc/blob/main/benchmarks/google_clusterdata_2019/conversion.example.yaml)
复制到仓库外。用准确 CSV 字节的 SHA-256 替换全零占位符，记录实际 UTC 导出时刻，审查
全部映射，并在私有位置或外部不可变存储中保留源 CSV。

```bash
climadc trace convert-google-v3 google-v3-a.csv conversion.yaml google-v3-a-conversion \
  --query-sql executed-export.sql
climadc trace verify-google-v3 google-v3-a-conversion --source-csv google-v3-a.csv
```

转换目录包含 `conversion-config.yaml`、`conversion-manifest.json`、`export-query.sql`、
`workload.csv` 和 `checksums.sha256`。发布是原子的，且拒绝覆盖已有路径。不传
`--source-csv` 时会验证成员关系、校验和、配置/查询/清单一致性及标准负载契约；传入源文件
后还会逐字节重做转换。

## 映射语义与限制

| 标准字段 | v3 输入或假设 | 证据边界 |
|---|---|---|
| `release_time`、`available_at` | submit 时间加声明的 UTC 场景 epoch | epoch 是场景映射，不是 Google 真实墙钟时间 |
| `energy` | 请求的归一化 CPU × 声明的 kW/CPU × 观测运行时长 × 利用率 | 场景估算，不是任务或设施实测能耗 |
| `max_power` | 请求的归一化 CPU × 声明的 kW/CPU | 场景估算；归一化 CPU 不是物理核数 |
| `deadline` | submit + 观测运行时长 × 声明倍数 | 构造场景时使用未来 finish 事件 |
| `preemptible` | 明确设为 `true` 的假设 | 不是轨迹事实 |
| `priority` | submit 事件 priority | 保留源字段 |

相对提交时刻，观测完成属于未来信息。转换器会把这一事实写入清单，并仅用它构造事后场景；
不得声称原调度器当时知道该 deadline 或 energy。因果 E2 Benchmark 必须把它论证为预声明的
场景抽象并进行敏感性分析，或换成历史时点真实可用的 deadline/runtime 信息。

## E2 仍缺什么

通过验证的转换包只满足工作负载来源门禁的一部分。E2 仍需要：

- 获得许可、不可变保留且正确署名的源数据；
- 带真实可用时间的前瞻采集或历史归档 forecast vintage；
- 预先声明外部负载时间线与气候/电网场景的关系；
- 相互独立的训练、校准与评测日期、站点或负载切片；
- 可辩护且经过敏感性分析的功率、deadline/slack 与 preemptibility 映射；
- 完整回放产物及哈希绑定的 claim registry 条目。

Google 工作负载/功率轨迹不得描述为同站点伦敦历史。全部门禁通过前，任何转换或下游回放都
不能标记为 E2 结果。
