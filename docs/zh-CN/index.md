# ClimaDC

ClimaDC 是一个防信息泄漏、契约优先的框架，用于气候感知的数据中心预测与影子决策评估。

它可以校验气象/DC 时间语义，运行时间回测基线，校准不确定性，对比能量守恒的影子决策，并发布八个可审计研究产物。Alpha 是离线研究框架，不是在线控制器，也不代表真实节能效果。

```mermaid
flowchart LR
    A["气象 + DC 输入"] --> B["标准契约"]
    B --> C["LeakageGuard"]
    C --> D["时间回测"]
    D --> E["校准 + 评估"]
    E --> F["影子决策"]
    F --> G["可审计产物"]
```

从[五条命令快速开始](quickstart.md)入门，再阅读 [`issue_time` / `available_at` / `valid_time` 指南](concepts/time-semantics.md)。WeatherDC 示例严格区分完全离线的合成 Benchmark 与经过校验的上游数据转换模式。

当前未发布的开发版本还加入了完整的 [v0.2 工程回放](design/v0.2-engineering-replay.md)：严格的电网信号、可延迟任务契约、本地读取器、六种默认带约束策略和可选的上分位数策略、单窗口或滚动执行、带已提交时段事后边际风险诊断的真实值结算、[多场景稳健性/Pareto 套件](concepts/robustness-suites.md)、带哈希输入和自包含报告的[英国参考回放](concepts/reference-replay.md)，以及[只读 Prometheus/Kepler、Carbon Aware SDK 和 SustainDC 适配器](concepts/read-only-integrations.md)。

## 当前边界

已实现本地 CSV/Parquet、可选 Xarray、当前/历史 Open-Meteo、NESO 英国全国碳强度与
WeatherDC 转换适配器。WeatherDC 完整路径只转换 HII 观测与电表数据，不虚构历史预报
可用时间、工作负载、控制数据或完整重训练结果。可选生态适配器只消费导出和 API 响应，
不会部署、配置或控制上游系统。
