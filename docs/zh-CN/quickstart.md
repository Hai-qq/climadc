# 快速开始

## 前置条件

- Python 3.10、3.11、3.12 或 3.13。
- 下方六行路径需要 Bash 或 Zsh；Windows 用户可使用 PowerShell 版本。
- 安装已发布的 Alpha（`python -m pip install "climadc==0.3.0a1"`）或源码检出（`python -m pip install -e .`）。

内置研究使用确定性、项目自有的合成数据，不需要 API Key 或网络连接。

## 可独立验证的快速路径

```bash
export CLIMADC_STUDY="$(mktemp -d)/climadc-quickstart"
climadc init "$CLIMADC_STUDY"
climadc validate "$CLIMADC_STUDY/study.yaml"
climadc benchmark "$CLIMADC_STUDY/study.yaml"
climadc verify-run "$CLIMADC_STUDY/runs/latest"
climadc report "$CLIMADC_STUDY/runs/latest"
```

最后一条命令会打印 `report.html` 的绝对路径。Windows PowerShell 可运行：

```powershell
$Study = Join-Path ([System.IO.Path]::GetTempPath()) ("climadc-" + [guid]::NewGuid())
climadc init $Study
climadc validate (Join-Path $Study "study.yaml")
climadc benchmark (Join-Path $Study "study.yaml")
climadc verify-run (Join-Path $Study "runs/latest")
climadc report (Join-Path $Study "runs/latest")
```

## 发布产物

`runs/latest` 指向不可变 schema v2 运行目录。`run-manifest.json` 声明完整文件集合；当前
Benchmark payload 包含 `run.yaml`、`lineage.json`、`splits.parquet`、`predictions.parquet`、
`metrics.json`、`leakage-report.json`、`dataset-card.md` 和 `report.html`，证据外层另含
`environment.json` 与 `checksums.sha256`。应使用 `verify-run` 验证契约，而不是断言固定文件数。

合成 Quickstart 只验证执行、契约与产物完整性，不代表真实站点预测精度或节能效果。

## 工程回放演示

v0.3 Alpha 还打包了带哈希校验的英国参考回放。它使用仓库内 Open-Meteo/NESO 衍生快照，以及明确声明的 UTC 电价、负载和 PUE 模型，因此运行过程完全离线：

```bash
climadc demo carbon-shift --output-dir ./climadc-replay-runs
climadc verify-run ./climadc-replay-runs/latest
climadc report ./climadc-replay-runs/latest
```

该路径会发布一套 schema v2 回放记录；文件集合由清单声明，不把固定数量作为长期 API。来源时间假设、可选联网刷新和声明边界见[英国参考回放案例](concepts/reference-replay.md)。

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

## 多场景敏感性套件

完全离线运行打包的四场景策略敏感性研究：

```bash
climadc demo sensitivity-suite --output-dir ./climadc-replay-suite-runs
climadc verify-suite ./climadc-replay-suite-runs/latest
climadc report ./climadc-replay-suite-runs/latest
```

自定义矩阵时，创建一份 `suite.yaml`，让每个场景指向完整的回放研究 YAML，再运行 `climadc replay-suite ./suite.yaml`。套件发布 v2 清单/校验和、机器可读场景行、等权敏感性汇总、Pareto 前沿、自包含报告，以及每场景一套可独立验证的 v2 子运行。旧 `demo robustness-suite` 仅为带弃用提示的兼容别名；只有真实改变声明日期、季节、位置或负载维度的矩阵才能使用 `suite_type: robustness`。详见[回放套件](concepts/robustness-suites.md)。
