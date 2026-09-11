# Design — Filtered Mean-Reversion Strategy

## State shape

```python
@dataclass
class FilteredMRState:
    """filtered_mr 持久状态: 大周期桶 + 增量指标 + 触轨 FSM"""

    # ---- 大周期桶跟踪 (用 bucket_ts_encoded 判定桶切换) ----
    higher_bucket_ts: int = 0               # 当前大周期桶 ts (整型, bucket_ts_encoded 输出)
    higher_close: float = 0.0              # 当前大周期已累积 close (最后一根)
    higher_high: float = 0.0               # 当前大周期 max high
    higher_low: float = 0.0                # 当前大周期 min low
    higher_count: int = 0                  # 当前大周期已累积 bar 数
    higher_ema_state: EMAState             # 大周期 close EMA 增量 state
    higher_has_ema: bool = False           # 大周期 EMA 是否就绪
    higher_last_close: float = 0.0         # 上一大周期 finalized close

    # ---- 本周期 EMA 通道 ----
    cur_ema_state: EMAState
    cur_count: int = 0

    # ---- ATR (Wilder smoothing) ----
    atr: float = 0.0
    atr_prev_close: float = 0.0
    atr_count: int = 0
    atr_window: list                        # 长度 = atr_ma_period; 用于 MA

    # ---- ADX (Wilder smoothing of +DM/-DM/TR/DX) ----
    plus_dm: float = 0.0
    minus_dm: float = 0.0
    tr_smooth: float = 0.0
    adx: float = 0.0
    adx_count: int = 0

    # ---- FSM: 上一桶触轨 ----
    touched_lower_prev: bool = False
    touched_upper_prev: bool = False
```

所有字段都是 dataclass 字段，符合 `test_no_instance_state_in_strategies` 正则约束。

## Step 流程

```text
def step(state, bar, params):
    ts_int = int(bar["ts"])
    higher_p_sec = resolve_period_seconds(params["higher_period"])

    # 1) 大周期桶切换检测
    new_higher_ts = bucket_ts_encoded(ts_int, higher_p_sec)
    if state.higher_count > 0 and new_higher_ts != state.higher_bucket_ts:
        # 旧桶 finalize: 用 state.higher_close 推进 higher EMA
        state.higher_ema_state, _ = ema_step(
            state.higher_ema_state, state.higher_close, params["higher_ema_period"])
        if state.higher_ema_state.count >= params["higher_ema_period"]:
            state.higher_has_ema = True
        state.higher_last_close = state.higher_close
    state.higher_bucket_ts = new_higher_ts
    if state.higher_count == 0 or new_higher_ts != state.higher_bucket_ts:  # new bucket
        state.higher_open = bar["o"]; state.higher_high = bar["h"]; ...
    state.higher_count += 1
    state.higher_close = float(bar["c"])
    state.higher_high = max(state.higher_high, float(bar["h"]))
    state.higher_low = min(state.higher_low, float(bar["l"]))

    # 2) 本周期 EMA + ATR + ADX 增量更新 (mark=0 也累积)
    state.cur_ema_state, _ = ema_step(state.cur_ema_state, float(bar["c"]), params["cur_ema_period"])
    state = _update_atr(state, bar, params)
    state = _update_adx(state, bar, params)

    if bar["mark"] == 0:
        return state, 0

    # 3) 过滤矩阵
    is_strong_trend = state.adx > params["adx_threshold"] and state.adx_count > 0
    is_high_vol = state.atr > params["atr_vol_mult"] * state.atr_ma and state.atr_count > params["atr_ma_period"]
    higher_trend = 0
    if state.higher_has_ema and state.higher_ema_state.ema > 0:
        higher_trend = 1 if state.higher_last_close > state.higher_ema_state.ema else -1

    # 4) 通道计算
    if state.atr <= 0 or state.cur_ema_state.count < params["cur_ema_period"]:
        state.touched_lower_prev = False
        state.touched_upper_prev = False
        return state, 0
    upper = state.cur_ema_state.ema + params["band_mult"] * state.atr
    lower = state.cur_ema_state.ema - params["band_mult"] * state.atr

    cur_low, cur_high, cur_close = float(bar["l"]), float(bar["h"]), float(bar["c"])
    touched_lower = cur_low < lower
    touched_upper = cur_high > upper
    close_in_lower = cur_close > lower   # 回到带内 (反转确认)
    close_in_upper = cur_close < upper

    sig = 0
    if state.touched_lower_prev and touched_lower and close_in_lower:
        if not is_strong_trend and not is_high_vol:
            if higher_trend >= 0:           # 大周期多头或震荡 → 允许做多
                sig = 1
    elif state.touched_upper_prev and touched_upper and close_in_upper:
        if not is_strong_trend and not is_high_vol:
            if higher_trend <= 0:           # 大周期空头或震荡 → 允许做空
                sig = -1

    state.touched_lower_prev = touched_lower
    state.touched_upper_prev = touched_upper
    return state, sig
```

## 关键算法

### ATR Wilder smoothing

```
TR_t = max(H - L, |H - prev_close|, |L - prev_close|)
ATR_t = ATR_{t-1} + (TR_t - ATR_{t-1}) / period
```

### ADX Wilder smoothing

```
+DM_t = max(H_t - H_{t-1}, 0) if (H_t - H_{t-1}) > (L_{t-1} - L_t) else 0
-DM_t = max(L_{t-1} - L_t, 0) if (L_{t-1} - L_t) > (H_t - H_{t-1}) else 0

TR_smooth_t = TR_smooth_{t-1} + (TR_t - TR_smooth_{t-1}) / period
+DM_smooth_t = +DM_smooth_{t-1} + (+DM_t - +DM_smooth_{t-1}) / period
-DM_smooth_t = -DM_smooth_{t-1} + (-DM_t - -DM_smooth_{t-1}) / period

+DI = 100 * +DM_smooth / TR_smooth
-DI = 100 * -DM_smooth / TR_smooth
DX = 100 * |+DI - -DI| / (+DI + -DI)   if (+DI + -DI) > 0 else 0

ADX_t = ADX_{t-1} + (DX_t - ADX_{t-1}) / period
```

`ADX` 就绪需要 `adx_count >= 2 * adx_period` (Wilder smoothing 标准)。

## params_spec

```python
params_spec = {
    "higher_period":    {"default": "1h", "type": str, "min": None, "max": None},  # 周期字符串
    "higher_ema_period": {"default": 60, "type": int, "min": 2, "max": 1000},
    "cur_ema_period":    {"default": 20, "type": int, "min": 2, "max": 1000},
    "atr_period":        {"default": 14, "type": int, "min": 2, "max": 1000},
    "atr_ma_period":     {"default": 20, "type": int, "min": 2, "max": 1000},
    "atr_vol_mult":      {"default": 1.5, "type": float, "min": 1.0, "max": 5.0},
    "adx_period":        {"default": 14, "type": int, "min": 2, "max": 1000},
    "adx_threshold":     {"default": 30.0, "type": float, "min": 10.0, "max": 100.0},
    "band_mult":         {"default": 1.8, "type": float, "min": 0.5, "max": 10.0},
}
```

## 不实现 batched_step

FSM latch (touched_lower_prev / touched_upper_prev) 跟 channel_deviation 类似，难以向量化。sweep 自动走 ThreadPool（spec Scenario "Hook default 未实现走 ThreadPool"）。

## 测试

`tests/test_filtered_mr.py`:
- `test_init_state_returns_dataclass`
- `test_filtered_mr_step_basic` (mark=0 早返回, sig 范围)
- `test_filtered_mr_strong_trend_filter` (ADX 高停所有信号)
- `test_filtered_mr_high_vol_filter` (ATR > mult * MA 停所有信号)
- `test_filtered_mr_higher_period_filter` (大周期多头禁做空, 空头禁做多)
- `test_filtered_mr_close_confirmation` (必须 2 桶才出信号)
- `test_filtered_mr_no_instance_state` (正则扫描)

## Files

修改：
- `openspec/specs/evtrade-architecture/spec.md` (brief mention, R1 加一个新策略说明)
- `kbs/06-交易策略详解.md` (新增 §filtered_mr 章节)
- `kbs/12-重构与性能内核.md` (perf note)

新增：
- `evtrade/strategies/filtered_mr.py`
- `evtrade/strategies/__init__.py` (无变化; register_strategy 自注册)
- `tests/test_filtered_mr.py`
