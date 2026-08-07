# 工程输入契约

尚未发布的 v0.2 语义骨架新增了严格的电网信号和可延迟批任务输入。契约从 `climadc.contracts` 导出，本地 CSV/Parquet 读取器从 `climadc.adapters` 导出。旧版 v0.1 影子调度器尚未消费这些新契约。

## 电网信号

`GridSignalFrame` 必须且只能包含以下列：

| 列 | 含义 |
|---|---|
| `site_id` | ClimaDC 站点标识 |
| `region_id` | 电网区域标识，例如 `GB-13` |
| `issue_time` | 预测起报时间；事后值为空 |
| `available_at` | 该值实际可供研究使用的时间 |
| `valid_time` | 该值对应的时间间隔 |
| `signal` | `carbon_intensity` 或 `energy_price` |
| `value` | 有限数值 |
| `unit` | 排放/能量或受支持货币/能量单位 |
| `source` | 数据提供方或 fixture 标识 |
| `quality` | `forecast`、`observed` 或 `estimated` |
| `quantile` | 可选预测分位数，必须严格位于 `(0, 1)` |

预测行必须满足 `issue_time <= available_at <= valid_time`。观测和估算行的 `issue_time`、`quantile` 必须为空，并满足 `valid_time <= available_at`。碳强度不得为负。

碳强度单位必须与 `gCO2e/kWh` 量纲相容，例如 `kgCO2e/MWh`。电价目前支持 GBP、USD、EUR 和 CNY 与相容能量单位的组合；框架不会隐式换汇。

分位数行与点预测始终分开。如果回放配置声明 `risk_quantile=q`，`ClimateForecastFrame` 的温度以及 `GridSignalFrame` 的电价、碳强度必须在每个所需规划时段都有精确的 `q` 行，并且在对应决策点因果可用。回放会拒绝不完整或有歧义的分位数场景，不做插值，也不退回点预测。这些输入行本身只定义上分位数压力场景，不代表区间覆盖率或联合概率声明。事后结算完成后，启用风险策略的研究会针对已提交时段发布描述性的边际覆盖率与损失诊断；它们不会改变输入语义，也不能证明联合校准。

```python
from pathlib import Path

from climadc.adapters import read_grid_signals

grid = read_grid_signals(
    Path("grid-signals.csv"),
    "csv",
    column_map={},  # 文件已经使用上面的标准列
    timezone="UTC",
)
grid_frame = grid.to_pandas()
```

源数据列名不同时，将每个源列映射到标准目标列：

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

映射后仍必须恰好产生全部标准列。提供方适配器负责把一个源记录展开为相互独立的预测行和事后行。已实现的 Prometheus/Kepler、Carbon Aware SDK 兼容和 SustainDC 转换器见[只读生态适配器](read-only-integrations.md)。

## 可延迟任务

`FlexibleWorkloadFrame` 每行表示一个任务，并且必须且只能包含以下列：

| 列 | 含义 |
|---|---|
| `job_id` | 非空任务标识，在站点内唯一 |
| `site_id` | ClimaDC 站点标识 |
| `release_time` | 物理上最早可执行时间 |
| `available_at` | 调度器获知任务的时间 |
| `deadline` | 最晚完成时间 |
| `energy` | 正的 IT 能量需求 |
| `energy_unit` | 与 `kWh` 量纲相容的单位 |
| `max_power` | 正的单任务 IT 功率上限 |
| `power_unit` | 与 `kW` 量纲相容的单位 |
| `preemptible` | 当前必须为 `true` |
| `priority` | 有限、非负的场景优先级 |

契约要求 `release_time <= available_at <= deadline`，并检查连续执行下界 `energy / max_power` 是否不超过从可用时间到截止时间的窗口。因此，在考虑站点容量竞争之前就能发现必然无法完成的任务。

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

## 时间戳与数据归属规则

- 只有本地读取器会依据声明的 IANA 时区解释无时区时间戳；
- 直接调用契约构造器时，输入必须带时区，并统一为严格 UTC；
- `to_pandas()` 默认返回深拷贝；
- 本地读取器负责校验列、时间、数值和单位，但不会虚构缺失的起报时间、获取时间或任务功率上限。

[工程回放内核](replay-kernel.md)直接消费这些契约，运行单站点的一个完整决策窗口或滚动序列。打包输入、来源时间边界和 12 文件发布契约见[英国参考回放案例](reference-replay.md)，整体边界见 [v0.2 工程回放技术设计](../design/v0.2-engineering-replay.md)。
