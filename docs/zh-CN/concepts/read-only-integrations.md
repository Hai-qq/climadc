# 只读生态适配器

阶段 3 把外部监控和仿真结果接入工程回放使用的同一套标准契约。适配器只读取响应或文件，不部署采集器、不修改负载、不调用控制接口，也不把上游平台变成 ClimaDC 的运行依赖。

## Prometheus 与 Kepler

`PrometheusRangeAdapter` 接收 Prometheus `query_range` 的标准 matrix 响应；`KeplerPrometheusAdapter` 在此基础上查询新版 [Kepler 指标清单](https://github.com/sustainable-computing-io/kepler/blob/main/docs/user/metrics.md)中的 `*_watts` gauge。

```python
import pandas as pd

from climadc.adapters import KeplerPrometheusAdapter

result = KeplerPrometheusAdapter().fetch_power(
    base_url="http://prometheus:9090",
    scope="node",
    component="cpu",
    start=pd.Timestamp("2026-08-01T00:00:00Z"),
    end=pd.Timestamp("2026-08-01T01:00:00Z"),
    step=pd.Timedelta(minutes=5),
    site_id="dc-1",
)
telemetry = result.telemetry
```

请求只使用 [Prometheus HTTP API](https://prometheus.io/docs/prometheus/latest/querying/api/) 的读取接口。样本时间写入 `event_time`，HTTP 响应完成时间写入 `available_at`。Kepler 默认标为 `estimated`，因为其上游值可能混合硬件测量和模型归因；只有来源能证明时才应显式使用 `quality="observed"`。

支持 node、container、pod、process，以及 CPU 虚拟机功率；Kepler 明确提供 GPU watts gauge 的层级也支持 GPU。适配器用显式 `sum by (...)` 聚合 zone 维度，在 `device_id` 中保留资源标识，并拒绝负功率。它不对累计 joule counter 做隐式差分；已有自定义聚合 PromQL 时可直接使用 `PrometheusRangeAdapter`。

`base_url` 禁止嵌入凭据，因为 URL 会进入谱系元数据。需要认证头或读取导出 JSON 时，应注入自定义 transport。

## Carbon Aware SDK 兼容接口

`CarbonAwareSDKAdapter` 消费 Green Software Foundation [Carbon Aware SDK Web API](https://github.com/Green-Software-Foundation/carbon-aware-sdk/blob/dev/casdk-docs/docs/tutorial-basics/carbon-aware-webapi.md) 的 DTO。适配器不绑定具体电网提供方；底层来源和凭据由用户配置的 Carbon Aware SDK 实例负责。

```python
import pandas as pd

from climadc.adapters import CarbonAwareSDKAdapter

adapter = CarbonAwareSDKAdapter()
forecast = adapter.fetch_current_forecast(
    base_url="http://carbon-aware:8080",
    location="eastus",
    site_id="dc-1",
    start=pd.Timestamp("2026-08-01T01:00:00Z"),
    end=pd.Timestamp("2026-08-01T06:00:00Z"),
)
```

预测转换器保留 `generatedAt` 作为 `issue_time`，并把真实响应获取时间作为 `available_at`；响应到达时已经过期的预测点会被拒绝。`fetch_observed` 单独读取结算数据，默认标为 `estimated`，避免把所有提供方结果都宣称成计量观测。数值遵循 SDK 声明的 `gCO2e/kWh`。

对于历史响应或需要外部认证的导出，可使用纯转换方法 `forecast_from_payload` 和 `observed_from_payload`；它们完全离线。适配器不会发送 SDK 的历史批量 POST 请求。

## SustainDC 评估导出

`SustainDCAdapter` 读取 [SustainDC](https://github.com/HewlettPackard/dc-rl) 官方生成的 `all_agents_episode_*.csv`，不依赖 Gymnasium、PyTorch 或 SustainDC 运行时。

```python
from pathlib import Path

import pandas as pd

from climadc.adapters import SustainDCAdapter

result = SustainDCAdapter().read_evaluation(
    Path("evaluation_data/all_agents_episode_1.csv"),
    site_id="sim-dc",
    region_id="sim-grid",
    start_time=pd.Timestamp("2026-08-01T00:00:00Z"),
    interval=pd.Timedelta(minutes=15),
)
```

| SustainDC 字段 | ClimaDC 输出 | 单位与质量 |
|---|---|---|
| `dc_ITE_total_power_kW` | 遥测 `it_power` | `kW`，估算 |
| `dc_HVAC_total_power_kW` | 遥测 `cooling_power` | `kW`，估算 |
| `dc_total_power_kW` | 遥测 `total_power` | `kW`，估算 |
| `outside_temp` | 遥测 `air_temperature` | `degC`，估算 |
| `bat_avg_CI` | 事后电网 `carbon_intensity` | `gCO2e/kWh`，估算 |

调用方必须给出绝对 UTC 起点和时长。适配器会校验上游 `day`/`hour` 唯一、有序且与声明间隔一致；每个数值在仿真间隔结束时才视为可用。归一化移峰字段不会被伪造成 `FlexibleWorkloadFrame`，因为评估导出没有逐任务释放、截止、能量和功率约束。

## 共同边界

三个适配器都返回不可变标准契约和机器可读元数据。默认测试只使用注入响应或本地 DataFrame，不访问网络。它们的作用是让外部证据进入离线研究，不会把 ClimaDC 变成遥测采集器、仿真器或生产调度器。
