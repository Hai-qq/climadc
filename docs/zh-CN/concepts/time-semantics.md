# 时间语义

ClimaDC 明确区分三个问题：

- `issue_time`：预报由模型或数据提供方生成的时刻；
- `available_at`：下游实验实际能够读取这条数据的时刻；
- `valid_time`：被预测物理状态对应的时刻。

气象预报必须满足 `issue_time <= available_at <= valid_time`。在决策时刻 `T`，只有 `available_at <= T` 的特征可用；仅比较 `issue_time` 或 `valid_time` 不足以防止泄漏。

## 合法示例

08:00 起报、08:07 到达、预测 12:00 状态的预报，可以用于 09:00 决策。

## 非法延迟到达示例

08:00 起报、09:20 才回填、预测 12:00 状态的预报，不能用于 09:00 决策。较早的 `issue_time` 不能让数据提前可用，`LeakageGuard` 会拒绝它。

## 非法“观测冒充预报”示例

12:00 的气象观测可以诚实记录为 `issue_time = available_at = valid_time = 12:00`，但不能把 `issue_time` 或 `available_at` 改成 09:00 来冒充提前预报。因此 WeatherDC 完整路径只做转换：上游 HII 行是观测，没有历史起报或获取时间。

遥测使用 `event_time`，并要求 `event_time <= available_at`。训练标签只有在训练起点前已经可用才是因果合法的；测试结束后的评价可以读取最终观测目标，但模型拟合不能读取它。

内部时间统一为带时区 UTC。没有时区的时间戳会被拒绝，除非适配器得到明确源时区并据此完成本地化。
