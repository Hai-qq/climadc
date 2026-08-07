# 回放稳健性套件

回放套件把多个可独立复现的回放研究汇总成跨场景策略比较。它复用既有求解器和每场景
12 项产物契约；不会自动篡改输入、虚构不确定性分布，也不会在调度前混合不同场景记录。

## 配置套件

每个场景都指向一份完整的回放 `study.yaml`，并保留自己的来源清单、假设和输入哈希：

```yaml
schema_version: "1"
suite_id: facility-policy-sensitivity
aggregation: equal_weight
scenarios:
  - scenario_id: base
    description: 声明的基础费率与目标权重
    study: scenarios/base/study.yaml
  - scenario_id: high-demand-charge
    description: 同一时域下声明的需量费用压力场景
    study: scenarios/high-demand-charge/study.yaml
assumptions:
  purpose: 检查策略对声明运行假设的敏感性
limitations:
  - 场景权重不是发生概率。
output_dir: replay-suite-runs
```

套件至少需要两份不同的研究文件，场景 ID 必须唯一且可安全用于目录名。求解前，ClimaDC
要求所有场景具有相同的时域、间隔、单窗口/滚动模式、滚动结构，以及是否提供可选风险策略；
求解后还要求币种一致、策略顺序一致。决策日期、站点、输入、目标权重、设施模型和声明费率
可以变化，但这些有符号变化是否具有科学可比性，仍由研究设计者负责。

运行自定义套件并定位报告：

```bash
climadc replay-suite ./suite.yaml --output-dir ./replay-suite-runs
climadc report ./replay-suite-runs/latest
```

公开 Python 接口使用同一套实现：

```python
from pathlib import Path

from climadc.replay import (
    ReplaySuiteArtifactWriter,
    ReplaySuiteConfig,
    ReplaySuiteRunner,
)

config = ReplaySuiteConfig.from_yaml(Path("suite.yaml"))
result = ReplaySuiteRunner().run(config)
run_path = ReplaySuiteArtifactWriter().write(result, config.output_dir)
print(run_path)
```

安装包还提供完全离线的四场景敏感性示例：

```bash
climadc demo robustness-suite --output-dir ./climadc-replay-suite-runs
climadc report ./climadc-replay-suite-runs/latest
```

示例复用同一份经过校验的英国快照，只改变联合目标权重与一个合成需量费用。它用于演示
套件机制，属于敏感性分析，不是样本外验证。

## 汇总语义

费用、碳排和峰值先在各自场景内计算相对 ASAP 的有符号变化：负数表示降低，正数表示增加。
套件针对每种策略报告：

- 场景总数、可行场景数和可行比例；
- 在可行场景中，有符号变化小于该指标单位下 `-1e-9` 的比例；
- 对可行场景进行等权算术平均后的变化；
- 可行场景中的最坏变化，即最大有符号变化，以及对应场景 ID；
- 基于平均费用、碳排和峰值变化的 Pareto 成员资格。

不可行策略仍会显示，但不会向均值或改善率贡献数值，也绝不会进入 Pareto 比较。只有在全部
声明场景中都可行的策略才有 Pareto 资格，并以 `1e-9` 相对数值容差同时最小化三个等权算术平均变化。它不是标量排名、
全局最优、概率加权风险度量、CVaR 或生产保证。

## 发布证据

每个不可变套件运行恰好包含八个顶层条目：

| 产物 | 用途 |
|---|---|
| `suite.yaml` | 与本地路径无关的汇总规则、兼容性、场景哈希、假设和限制 |
| `lineage.json` | 套件运行 ID、软件版本、时间戳和场景配置哈希 |
| `scenario-index.json` | 场景元数据、可行性、输入哈希和相对子运行路径 |
| `scenario-metrics.parquet` | 每个场景—策略的一行状态/结算记录 |
| `robustness-metrics.json` | 等权可行性、改善率、均值和最坏情形汇总 |
| `pareto-frontier.json` | 资格规则、目标、单位和非支配策略 |
| `report.html` | 自包含对比报告，并链接本地场景报告 |
| `scenarios/` | 每个场景一套经过独立校验的完整 12 项回放产物 |

发布过程是原子的。只有所有汇总文件和所有场景子运行都通过校验后，套件级 `latest` 才会更新。
不可变套件目录内，每个场景也保留自己的相对 `latest` 指针。

## 声明边界

等权是明确的分析约定，不是估计出的发生概率。即使使用场景内基线变化，不同负载规模或站点的
均值仍可能被大场景主导。“最坏”只表示有限声明场景中的最坏值，不是尾部风险界。因而，有意义
的稳健性结论还需要充分的场景设计、来源与覆盖证据，不能只靠增加 YAML 文件数量。
