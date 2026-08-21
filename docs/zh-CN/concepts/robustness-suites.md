# 回放敏感性与稳健性套件

回放套件运行多份完整、可独立验证的研究，并以各场景自己的 ASAP 为基线汇总有符号差值。
`suite_type` 限定可做出的表述：

- `sensitivity` 在同一样本内改变目标碳价、合成需量费等假设；
- `robustness` 必须声明并真实改变 `decision_date`、`season`、`location` 或 `workload`
  维度。数值不同本身仍不能证明统计独立，来源和覆盖范围继续限制结论。

内置套件复用同一个伦敦日期和负载，因此只能称为敏感性分析：

```bash
climadc demo sensitivity-suite --output-dir ./climadc-replay-suite-runs
climadc verify-suite ./climadc-replay-suite-runs/latest
climadc report ./climadc-replay-suite-runs/latest
```

旧 `demo robustness-suite` 只是带弃用提示的兼容别名，不代表样本外验证。

所有场景必须共享时域、间隔、单窗口/滚动形态、风险策略可用性、指标 schema、策略顺序和
币种。目标参数可以在 sensitivity 矩阵中变化。等权只是算术权重，不是概率；“最坏值”只
是有限声明场景内的最大值，不是尾部风险。

顶层 v2 产物包含套件配置、谱系/环境清单、递归校验和、场景索引、逐场景指标、
`suite-metrics.json`、Pareto 汇总、自包含报告与 `scenarios/`。每个子目录都是完整 v2
运行，`verify-suite` 会递归验证。具体文件集合由 `run-manifest.json` 决定，不以固定数量
作为长期 API。

套件汇总 Pareto 与单研究 `objective.mode: pareto_analysis` 不同：前者比较完整场景中的
策略平均差值，后者输出一个研究内所有预先声明的碳价点。即使 robustness 套件跨日期，
只要负载仍是合成数据，它也不会自动升级为 E2 或 E3。
