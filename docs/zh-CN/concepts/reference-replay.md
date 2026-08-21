# 英国参考回放案例

打包的 `gb-london-carbon-shift-24h` 是 **E1 单日机制演示**：仓库内
Open-Meteo/NESO 衍生信号与项目生成负载、明确的 UTC 电价、示意碳价及参考 PUE 模型组合。
它不证明生产节省、稳健性、边际避免排放或同站点运营效果。

```bash
climadc demo carbon-shift --output-dir ./climadc-replay-runs
climadc verify-run ./climadc-replay-runs/latest --json
climadc report ./climadc-replay-runs/latest
python benchmarks/reference/gb_london_24h/reproduce.py --check
```

精简 golden summary 与容差理由位于 `benchmarks/reference/gb_london_24h/`；README 的定量权衡
语句绑定到 `evidence/claims.yaml` 中的 `E1-LONDON-TRADEOFF-001`。

| 输入 | 决策用途 | 结算用途 | 边界 |
|---|---|---|---|
| Open-Meteo Previous Runs 温度 | 固定提前 24 小时预报 | 无 | 决策可用时间是场景假设 |
| Open-Meteo Historical Weather 温度 | 无 | 环境温度 | 网格估算，不是站点实测 |
| NESO 英国全国碳强度 | 预报 | provider estimated actual | 全国平均；每两个半小时按 UTC 小时平均 |
| 声明的 UTC 分时电价 | 预报电价 | 相同的场景结算 | 项目场景，不是供应商费率或账单 |
| 四个带截止时间任务 | 可调度负载 | 完成/SLA 核算 | 项目 fixture，不是生产轨迹 |

伦敦标签不等于英国全国碳区域。天气与碳结算继续标记为 `estimated`，报告不会简称为
observed actual；排放指标统一为 `estimated_location_based_emissions_kgco2e`。

schema v2 由 `run-manifest.json` 声明准确文件集合，`environment.json` 记录运行环境，
`checksums.sha256` 用稳定 POSIX 相对路径覆盖其他全部产物。回放专属文件包括可移植假设、
来源/谱系、标准 Parquet、调度、剖面、求解器状态、指标和离线 HTML。应使用
`verify-run` 重建并验证，而不是断言固定文件数。

联网刷新只创建新目录且拒绝覆盖。新快照会在解析前保存
`raw/openmeteo-forecast.json`、`raw/openmeteo-settlement.json`、`raw/neso-carbon.json`，
并用 `raw/retrieval-metadata.json` 记录公共请求、HTTP 状态、白名单响应头、raw 哈希、
解析器版本、变换、canonical 哈希、许可与署名；不会保存 token、cookie 或认证头。

仓库内 fixture 早于 raw capture，只能证明 canonical 文件哈希；缺失的 provider 原始字节
不会被补造。内置四场景复用该日期和负载，因此命令为 `demo sensitivity-suite`，详见
[套件语义](robustness-suites.md)。
