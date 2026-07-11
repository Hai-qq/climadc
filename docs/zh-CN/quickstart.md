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
