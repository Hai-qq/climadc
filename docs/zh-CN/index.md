# ClimaDC

ClimaDC 是一个面向气候感知数据中心决策、具备因果时间语义和可审计证据链的离线评测与回放框架。

它可以校验气象/DC 时间语义，运行时间回测基线，对比受约束影子决策，并发布版本化、可独立验证的证据记录。Alpha 不是在线控制器，也不代表真实节能效果。

```mermaid
flowchart LR
    A["气象 + DC 输入"] --> B["标准契约"]
    B --> C["LeakageGuard"]
    C --> D["时间回测"]
    D --> E["校准 + 评估"]
    E --> F["影子决策"]
    F --> G["可审计产物"]
```

从[可独立验证的快速开始](quickstart.md)入门，再阅读 [`issue_time` / `available_at` / `valid_time` 指南](concepts/time-semantics.md)。WeatherDC 示例严格区分完全离线的合成 Benchmark 与经过校验的上游数据转换模式。

v0.3 Alpha 新增[独立产物验证](../evidence-model.md)、[E0–E3 证据等级](../benchmark-evidence-levels.md)、量纲明确的目标函数、[敏感性/稳健性套件语义](concepts/robustness-suites.md)、精简可生成的[英国 E1 参考回放](concepts/reference-replay.md)及只读来源适配器。当前证据只达到 E0 与 E1。

## 当前边界

已实现本地 CSV/Parquet、可选 Xarray、当前/历史 Open-Meteo、NESO 英国全国碳强度与
WeatherDC 转换适配器。WeatherDC 完整路径只转换 HII 观测与电表数据，不虚构历史预报
可用时间、工作负载、控制数据或完整重训练结果。可选生态适配器只消费导出和 API 响应，
不会部署、配置或控制上游系统。
