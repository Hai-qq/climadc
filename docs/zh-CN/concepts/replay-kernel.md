# 工程回放内核

v0.3 Alpha 回放内核把标准工程输入转成带约束的反事实对比。它既可以求解单站点的一个完整决策窗口，也可以反复重算滚动时域，再用估算结算天气、电价和碳强度结算每个已提交方案；它不会向生产基础设施下发任务。

## 已实现内容

`ReplayEngine` 默认在完全相同的任务和 IT 容量约束下比较六种策略；设置 `risk_quantile` 后增加第七种：

| 策略 | 决策时使用的目标 |
|---|---|
| `asap` | 在最早的合法时段执行已知任务；发生争用时高优先级先执行，作为基线 |
| `peak` | 最小化预测站点峰值 |
| `price` | 最小化预测电量费用和配置的需量费用 |
| `carbon` | 最小化预测位置法运营排放 |
| `joint` | 最小化配置的版本化目标 |
| `risk_aware` | 使用声明的温度、电价、碳强度上分位数组合场景，最小化该目标 |
| `oracle` | 用结算输入最小化该目标，只作为后见对照 |

精确 UTC 决策时刻已经知道的任务都是硬约束：每个任务必须守恒其 IT 能量，并满足释放时间、截止时间、单任务最大功率和共享 IT 容量。`available_at` 晚于决策时刻的任务不进入优化，但会计入 `future_jobs`；已经到达的任务绝不会被静默丢弃。

优先级只负责安排 ASAP 争用时的先后顺序，不允许丢弃低优先级任务；每个策略仍必须完成全部已接收任务。

每个策略先求解其声明的目标；随后第二个线性规划固定每个时段的聚合功率，并按 ASAP 的优先级、截止时间、
释放时间和任务 ID 顺序把功率分配到任务。这个择优步骤不会改善或恶化原策略目标，也不会改变聚合功率；
当 HiGHS 存在多个等价任务分配时，它会让任务级调度和 `shifted_energy_kwh` 可跨平台重现。

参考 `TemperatureSensitivePUEModel` 使用有上下界的线性曲线：

```text
pue[t] = clip(
    base_pue + slope_per_degree_c * (temperature_c[t] - reference_temperature_c),
    min_pue,
    max_pue,
)
facility_power_kw[t] = pue[t] * (fixed_it_power_kw + flexible_it_power_kw[t])
```

`FacilityEnergyModel` 是公开协议，因此可直接替换为站点标定模型，不必改优化器。

## 最小 API 流程

先按[工程输入指南](engineering-inputs.md)构造四个严格契约，再运行一个决策窗口：

```python
import pandas as pd

from climadc.replay import ReplayConfig, ReplayEngine, TemperatureSensitivePUEModel

config = ReplayConfig(
    site_id="dc-1",
    horizon=pd.Timedelta(hours=24),
    interval=pd.Timedelta(hours=1),
    it_capacity_kw=500.0,
    fixed_it_power_kw=300.0,
    objective_mode="monetized",
    carbon_price_currency_per_tco2e=1000.0,
    demand_charge_per_kw=0.0,
)

result = ReplayEngine(TemperatureSensitivePUEModel()).run(
    decision_time=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
    climate_forecast=climate,
    actual_weather=weather,
    grid_signals=grid,
    workload=workload,
    config=config,
)

print(result.status)
print(result.metrics)
```

决策时刻必须已经是精确 UTC 的 `pandas.Timestamp`，且时域必须是间隔的整数倍。输入时间戳表示区间起点；只有当区间起点不早于任务释放时间、区间终点不晚于截止时间时，任务才能使用该时段。

研究 YAML 推荐版本化 `objective`：`monetized` 用碳价把 kgCO2e 除以 1000 后换算到币种；`epsilon_constraint` 在声明的决策依据排放/峰值上界下最小化费用；`pareto_analysis` 输出全部固定碳价点且不挑选最好结果，目前只支持单窗口。旧 `cost_weight` / `carbon_weight` 保留原算术但发出弃用警告，其分数是量纲不统一的比较值，不是货币或统一效用。迁移示例见 [v0.3 指南](../../migration-v0.3.md)。

研究运行器通过 `climadc replay` 暴露同一套引擎。在研究 YAML 中加入以下可选字段即可启用阶段 4 的两项能力：

```yaml
replay:
  # 其他既有回放字段……
  risk_quantile: 0.9
rolling:
  periods: 24
  step: 1h
```

`rolling.periods` 是决策点数量；`rolling.step` 必须是回放间隔的正整数倍，且不能大于完整时域。省略 `rolling` 就保持单窗口行为；省略 `risk_quantile` 就保持原来的六策略对比。

## 因果选择与单位

内核对每个必需时段选择在决策时刻之前已经起报且已经可用的最新点预报。优先使用空分位数；没有空分位数时接受中位数（`quantile=0.5`）。如果存在同样新的多个候选值，内核会明确报歧义错误，而不是依赖行顺序挑一个。

配置 `risk_quantile=q` 后，风险感知策略会在每个时段分别为温度、电价和碳强度选择最新且精确标记为 `q` 的预测，且 `q` 必须严格位于 `(0.5, 1)`。任何缺失、歧义或不同标签的分位数都会报错；内核不会插值，也不会退回点预测。把三个边际上分位数组合成一个声明的压力场景，并不等于联合概率保证、CVaR 或经过校准的双侧预测区间。

事后值单独选择，只用于结算和 Oracle 对照。内核把兼容单位统一为：

- 温度：`degC`；
- IT 能量和功率：`kWh`、`kW`；
- 碳强度：`kgCO2e/kWh`；
- 电价：同一种受支持币种的每 `kWh` 价格。

内核不做汇率换算；一次回放混用币种会报错。`demand_charge_per_kw` 使用同一币种，并乘以本决策窗口的事后峰值；它只是场景假设，不是对公用事业月度账单的完整复原。

## 事后上分位数诊断

配置 `risk_quantile=q` 且结算成功后，研究运行器会分别回测温度、电价和碳强度分位数。单窗口运行使用其已结算时段；滚动运行只使用实际提交的时段，以及每个决策点当时选择的分位数视图。规划过但未提交的时域行不会进入样本。

`replay-metrics.json` 和 `report.html` 会对每个边际信号发布：

- 包含等号的经验覆盖率，`actual <= quantile_forecast` 计为覆盖；
- 有符号覆盖差 `empirical_coverage - q`，负值表示覆盖不足；
- 经验覆盖比例的双侧 95% Wilson 二项区间；
- 标准信号单位下的超越次数、全时段平均正超越、发生超越时的条件均值和最大超越；
- 同一单位下、分位数为 `q` 的平均 pinball loss。

报告显式给出样本数，因为短回放的区间可能很宽。这些只是事后边际诊断：不会改变决策，不会拟合或重新校准分位数，不提供下界，不证明三个信号的联合 `q` 覆盖，也不会把压力场景变成 CVaR 目标。该策略只接收一个声明的上分位数，因此仍不具备双侧区间覆盖率。

## 结果表

`ReplayResult` 和 `RollingReplayResult` 都是冻结对象，每个 DataFrame 属性都会返回防御性副本：

- `status`：每种策略的求解可行性和消息；
- `allocations`：任务—时段级 IT 功率和能量；
- `profiles`：预测/实际温度、PUE、电网信号、IT 功率和站点功率；
- `metrics`：按事后值计算的 `kWh`、`kgCO2e`、费用、峰值 `kW`、SLA、转移能量、相对基线变化和 Oracle 后悔值；
- `violations`：决策窗口不可行时的明确原因。

研究级 `forecast_metrics` 还包含点预测 MAE，以及配置风险分位数时的上述上分位数诊断；同一 payload 会作为 `replay-metrics.json` 的 `forecast` 字段发布。

滚动结果还包含 `decisions`，记录每个决策点、每种策略的求解状态和已提交能量；`remaining_energy` 保存最终按策略区分的任务状态。调度和剖面中的 `decision_time` 可把每个已提交记录追溯到生成它的预测视图。版本化产物契约会记录滚动模式、决策次数、提交间隔和逐决策求解记录。

结算使用事后站点能耗：

```text
facility_energy_kwh = sum_t(actual_pue[t] * total_it_power_kw[t] * interval_hours)
estimated_location_based_emissions_kgco2e = sum_t(facility_energy_kwh[t] * estimated_settlement_carbon_kgco2e_per_kwh[t])
energy_charge = sum_t(facility_energy_kwh[t] * actual_energy_price[t])
demand_charge = actual_peak_kw * demand_charge_per_kw
```

`objective_regret` 等于某策略的结算目标减去 Oracle 目标。monetized 模式下单位是声明币种，legacy 模式下只是量纲不统一的分数。单窗口中该值非负；滚动模式下 Oracle 仍遵守任务 `available_at`，未来到达任务可能让累计有符号差值为负，因此它不是全局完美预见下界。费用、估算位置法排放、能耗和峰值始终分别保留。`shifted_energy_kwh` 是任务—时段分配相对 ASAP 的 L1 距离一半。

## 滚动行为与当前边界

`RollingReplayEngine` 采用滚动时域编排：每个决策点只选择当时因果可用的输入，对每种策略重算完整配置时域，只提交 `rolling.step`，并按策略分别结转任务剩余能量。总能耗、费用、碳排、峰值、SLA、转移能量和 Oracle 后悔值只统计已提交时段，避免后续重优化把尚未执行的计划能量重复计入。

阶段 2 运行器、历史 Open-Meteo/NESO 适配器、CLI、带哈希回放产物和对比 HTML 报告都支持该模式；打包的单窗口案例见[英国参考回放](reference-replay.md)。当前限制明确如下：

- 执行仍是单站点、离线且不下发控制；
- 任务在某个决策点被接收后，其截止时间必须落在该决策点的完整规划时域内；更长生命周期或需要终端结转的任务会被拒绝，而不是截断；
- 上分位数策略仍只是用户声明的压力场景，不是拟合后的分布风险模型或联合覆盖保证；输出的边际诊断仅供描述；
- 每次滚动求解会优化当前完整时域的需量费用项，最终结算则只对全部已提交时段的峰值计费一次；它不是针对完整月度账单峰值的动态规划。

回放中的事后值只是反事实结算输入，不能证明生产站点实际取得了同等收益。

如需在多个完整研究之间比较这些策略，同时保留场景级来源，请使用[回放敏感性/稳健性套件](robustness-suites.md)。套件只在本内核之上做编排与汇总，不改变求解语义。
