# 快速开始

## 前置条件

- Python 3.10、3.11、3.12 或 3.13。
- 下方五条命令需要 Bash 或 Zsh；Windows 用户可使用 PowerShell 版本。
- 先安装源码检出：`python -m pip install -e .`。Alpha 发布后，等价命令为 `python -m pip install "climadc==0.1.0a1"`。

内置研究使用确定性、项目自有的合成数据，不需要 API Key 或网络连接。

## 五条命令

```bash
export CLIMADC_STUDY="$(mktemp -d)/climadc-quickstart"
climadc init "$CLIMADC_STUDY"
climadc validate "$CLIMADC_STUDY/study.yaml"
climadc benchmark "$CLIMADC_STUDY/study.yaml"
climadc report "$CLIMADC_STUDY/runs/latest"
```

最后一条命令会打印 `report.html` 的绝对路径。Windows PowerShell 可运行：

```powershell
$Study = Join-Path ([System.IO.Path]::GetTempPath()) ("climadc-" + [guid]::NewGuid())
climadc init $Study
climadc validate (Join-Path $Study "study.yaml")
climadc benchmark (Join-Path $Study "study.yaml")
climadc report (Join-Path $Study "runs/latest")
```

## 发布产物

`runs/latest` 指向不可变运行目录，其中恰好包含 `run.yaml`、`lineage.json`、`splits.parquet`、`predictions.parquet`、`metrics.json`、`leakage-report.json`、`dataset-card.md` 和 `report.html`。

合成 Quickstart 只验证执行、契约与产物完整性，不代表真实站点预测精度或节能效果。

## 工程回放演示

v0.2 开发版本还打包了带哈希校验的英国参考回放。它使用仓库内 Open-Meteo/NESO 衍生快照，以及明确声明的电价、负载和 PUE 模型，因此运行过程完全离线：

```bash
climadc demo carbon-shift --output-dir ./climadc-replay-runs
climadc report ./climadc-replay-runs/latest
```

该路径会发布一套独立的 12 项回放记录。准确文件、来源时间假设、可选联网刷新和声明边界见[英国参考回放案例](concepts/reference-replay.md)。

## 滚动与上分位数回放

通用回放命令还支持在 `study.yaml` 中配置阶段 4 选项：

```yaml
replay:
  # 其他既有回放字段……
  risk_quantile: 0.9
rolling:
  periods: 24
  step: 1h
```

仍使用同一套 CLI：

```bash
climadc replay ./study.yaml --output-dir ./replay-runs
climadc report ./replay-runs/latest
```

打包的英国 fixture 没有声明分位数输入，因此仍是六策略、单窗口示例。启用风险策略的研究必须在每个规划时段提供精确且因果可用的同一分位数温度、电价和碳强度。滚动状态语义与当前截止时间限制见[工程回放内核](concepts/replay-kernel.md)。

启用 `risk_quantile` 后，同一命令还会把已提交时段的边际诊断写入 `replay-metrics.json` 和 `report.html`，包括经验覆盖率、95% Wilson 区间、覆盖差、超越幅度和 pinball loss；无需新增配置。这些诊断不会把三个边际场景变成联合概率保证。

## 多场景稳健性套件

完全离线运行打包的四场景策略敏感性研究：

```bash
climadc demo robustness-suite --output-dir ./climadc-replay-suite-runs
climadc report ./climadc-replay-suite-runs/latest
```

自定义矩阵时，创建一份 `suite.yaml`，让每个场景指向完整的回放研究 YAML，再运行 `climadc replay-suite ./suite.yaml`。套件发布八个顶层条目，包括机器可读的场景行、等权稳健性汇总、Pareto 前沿、自包含报告，以及每场景一套完整的 12 项子运行证据。兼容性门槛和声明限制见[回放稳健性套件](concepts/robustness-suites.md)。
