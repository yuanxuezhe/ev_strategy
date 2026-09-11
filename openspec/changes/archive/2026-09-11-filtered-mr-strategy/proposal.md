# Filtered Mean-Reversion Strategy (`filtered_mr`)

## Why

`channel_deviation` 是均值回归型策略，在单边暴涨暴跌行情中频繁接飞刀（逆势硬扛），需要 4 重过滤搭双重防护：
1. 大周期趋势过滤（顺势）
2. ADX 趋势强度过滤（避开单边）
3. ATR 波动率过滤（避开暴动）
4. Close 确认过滤（避免影线刺穿即开仓）

## What changes

### 新增

- `evtrade/strategies/filtered_mr.py`（新策略）
- 注册 `@register_strategy("filtered_mr")`
- State dataclass 含 大周期桶跟踪 + EMA/ADX/ATR 增量 + 触轨 FSM
- params_spec 包含 9 个参数（`higher_period`, `higher_ema_period`, `cur_ema_period`, `atr_period`, `atr_ma_period`, `atr_vol_mult`, `adx_period`, `adx_threshold`, `band_mult`）
- 测试 `tests/test_filtered_mr.py`
- spec brief mention
- KB §6 章节加一段

### 不变

- `VectorizedStrategy` 契约（step/init_state/format_signal_line）
- `bar` dict 契约 `{ts, o, h, l, c, v, mark}`
- `mark=0` 早返回语义
- `_execute_trades` / 引擎 / `reconcile`
- 其他策略（channel_deviation / ma_crossover）任何代码
- batched_step 不实现（FSM latch 跟 ma_crossover 同复杂度，独立评估）

## 策略逻辑

```
对每根当前周期 bar:
  1. 大周期桶跟踪 (--period 决定当前周期; --higher-period 决定大周期)
     - 桶切换: finalize 旧桶 close 入 higher EMA
  2. EMA / ATR / ADX 增量更新 (mark=0 也累积, 跟 ma_crossover 同语义)
  3. mark=0 早返回
  4. 通道计算: upper = cur_ema + band_mult * atr; lower = cur_ema - band_mult * atr
  5. 触轨检测: low < lower (下轨) / high > upper (上轨)
  6. 过滤矩阵:
     - 强趋势: adx > adx_threshold → 全停
     - 高波动: atr > atr_vol_mult * atr_ma → 全停
     - 大周期多头 (close > higher_ema) → 屏蔽上轨做空
     - 大周期空头 (close < higher_ema) → 屏蔽下轨做多
  7. Close 确认: 上一桶触轨 + 本桶 close 回到带内 → 才出信号
  8. 返回 (state, sig)
```

## Impact

- 新策略可跟现有策略一样用 `python -m evtrade {backtest,sweep,replay} --strategy filtered_mr ...`
- params 9 个，符合 `params_spec` 白名单 + 网格扫描规则
- State 复杂度高（大周期桶 + ADX/ATR 累加器 + FSM），但都在 `@dataclass` 内、不挂 `self`
- 不实现 `batched_step`（FSM 难向量化），sweep 自动走 ThreadPool
