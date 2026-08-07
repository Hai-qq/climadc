# 英国参考回放案例

打包的参考案例把 v0.2 契约和回放内核串成一个完整、离线的工程演示。它针对伦敦参考站点执行 24 小时、单决策点的反事实回放，不需要 API Key 或网络。

## 一条命令运行

```bash
climadc demo carbon-shift --output-dir ./climadc-replay-runs
climadc report ./climadc-replay-runs/latest
```

第一条命令打印不可变运行目录，第二条只打印静态报告路径，不会自动打开浏览器。同一组输入也可以走通用接口：

```bash
climadc replay path/to/study.yaml --output-dir ./replay-runs
```

安装包还会复用这份经过校验的快照，提供四场景的目标权重/需量费用敏感性演示。运行
`climadc demo robustness-suite`，并参阅[回放稳健性套件](robustness-suites.md)；新增场景只是
声明假设的变化，不是新的独立观测。

## Fixture 到底包含什么

| 输入 | 决策用途 | 结算用途 | 来源边界 |
|---|---|---|---|
| Open-Meteo [Previous Runs](https://open-meteo.com/en/docs/previous-runs-api) `temperature_2m_previous_day1` | 固定提前 24 小时的天气预报 | 无 | CC BY 4.0 API 数据；决策时刻可用性是场景假设 |
| Open-Meteo [Historical Weather](https://open-meteo.com/en/docs/historical-weather-api) `temperature_2m` | 无 | 环境温度 | 网格化模型/再分析估算，不是站点实测遥测 |
| [NESO Carbon Intensity API](https://api.carbonintensity.org.uk/) 英国全国信号 | 碳强度预报 | 事后估算值 | 每两个半小时值取小时均值；历史响应没有原始预报起报/可用时间，因此由场景声明 |
| 声明的分时电价 | 电价预报 | 数值相同的场景结算电价 | 项目生成场景，不是供应商费率或真实账单 |
| 四个带截止时间的任务 | 可调度负载 | 完成情况与 SLA 核算 | 项目生成的确定性 fixture，不是生产轨迹 |

站点标签是伦敦，但碳信号是英国全国值；该空间错配会明确写入来源清单和报告。有界温度敏感 PUE 是声明的参考关系，不是针对伦敦真实机房标定的模型。

## 可审计产物

每个不可变回放目录恰好包含 12 个文件：

| 产物 | 用途 |
|---|---|
| `assumptions.yaml` | 回放、站点模型、费率、目标权重和限制假设 |
| `source-manifest.yaml` | 来源 URL、获取时间、时间依据、许可证、变换，以及绑定到运行目录 Parquet 输入的 SHA-256 |
| `lineage.json` | 运行 ID、软件版本、配置哈希与原始已验证输入哈希 |
| `climate-forecast.parquet` | 调度策略在决策时刻可合法使用的天气预报 |
| `actual-weather.parquet` | 只用于事后结算的天气估算 |
| `grid-signals.parquet` | 严格区分质量和时间的预测/结算碳信号与电价 |
| `workload.parquet` | 标准化任务与服务约束 |
| `schedules.parquet` | 六种策略逐任务分配结果 |
| `profiles.parquet` | 每时段预测/事后 PUE、IT 功率、站点功率、电价和碳强度 |
| `solver-status.json` | 各策略可行性和求解器状态；滚动运行还包含逐决策记录与最终任务剩余能量 |
| `replay-metrics.json` | 预报误差、物理/经济/SLA 指标、相对基线变化与 Oracle 后悔值 |
| `report.html` | 内联 CSS、无脚本、无外部资源的自包含对比报告 |

HTML 中的每个对比数字都能从 Parquet 和 JSON 重建。程序会在解析前核对原始输入哈希并将其保留在谱系中，再把发布后的来源清单重新绑定到运行目录内的 Parquet，因此单独拷走该目录仍能复核。fixture 一旦被修改就停止运行，不会静默改变研究结果。

## 刷新快照

刷新是可选联网操作，只能创建新目录，拒绝覆盖已有路径：

```bash
climadc demo refresh-carbon-shift ./gb-snapshot --decision-date 2026-08-01
climadc replay ./gb-snapshot/study.yaml
```

刷新命令调用 Open-Meteo Previous Runs、Archive API 和 NESO Carbon Intensity API，再按同一套契约归一化并写入带哈希的来源清单。刷新不会“补造”历史接口里缺失的原始起报/可用时间；这些字段仍明确标为场景假设。

## 如何解读

某个策略可能改善一个指标、同时恶化另一个指标。报告保留所有带符号变化，不隐藏负收益；联合目标是加权比较分数，不是货币。由于费率、负载和站点模型属于场景，结果证明的是因果回放与工程核算能力，不是真实生产节能。
