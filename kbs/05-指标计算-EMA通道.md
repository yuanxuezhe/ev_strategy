# 05 指标计算：EMA 与通道轨

> 相关源码：`indicators/ema.py`（`evtrade/indicators/ema.py`）。
> 历史包袱：`evtrade/core/incremental_indicators.py` 已在 change `2026-09-09-decouple-indicators-from-framework` 中删除——framework 不再持有 EMA 状态，所有指标计算由策略通过 `evtrade/indicators/` 增量 API 在 DSL body 内自维护。

## 1. 指标定义：通达信蓝色通道轨

```
UP1 := EMA(H, TF1)    上轨（对周期桶最高价做 EMA）
DW1 := EMA(L, TF1)    下轨（对周期桶最低价做 EMA）
TF1 = 21（默认，由策略 params_spec 自定义）
```

即上轨是"最高价的 21 周期 EMA"、下轨是"最低价的 21 周期 EMA"，构成一条随价格平滑移动的通道。

## 2. EMA 数学定义（本实现的口径）

- 平滑系数：`k = 2/(p+1)`（p=21 时 k≈0.0909）
- **种子（seed）**：前 p 个值做简单平均 `SMA = sum(前p个)/p`，作为第一个 EMA
- 之后递推：`EMA_t = price_t * k + EMA_{t-1} * (1-k)`

`evtrade/indicators/ema.py` 提供**双接口**，口径一致：

| 实现 | 用途 | 复杂度 |
|---|---|---|
| `ema(values, p)` / `ema_channel(highs, lows, p)` 纯函数 | 复盘 / jupyter / 一次性批量 | O(n) |
| `@njit ema_push(state, value, p)` / `ema_current(state, pending, p)` 增量版 | **DSL body 热路径调用**（Python ref + numba kernel + CUDA device 三端同源） | O(1)/bar |
| `@njit ema_channel_push(...)` / `ema_channel_current(...)` 双 rail | | O(1)/bar |

## 3. 增量版：`ema_push` / `ema_current`（DSL 可调用）

增量 state 形状为 `(sum, count, ema)` 三元组，初始 `(0.0, 0, inf)`。

### `ema_push(s_sum, s_count, s_ema, value, p) -> (sum, count, ema)`

桶闭合 / 每根 bar 推入时调用，O(1)：

```
count <  p :  sum += value; count++; 若 count == p: ema = sum/p      ← SMA seed
count >= p :  ema = value*k + ema*(1-k); count++
```

返回新 state（按值，numba 端是 `UniTuple`）。

### `ema_current(s_sum, s_count, s_ema, p, pending) -> float`

带当前未闭合 bar 的"试探性 EMA"，O(1)，不改 state：

```
count <  p-1 :  返回 inf（数据不足）
count == p-1 :  返回 (sum + pending) / p        ← pending 凑满 p 个，得 SMA seed
count >= p   :  返回 pending*k + ema*(1-k)
```

numba 端无 `None`，用 `inf` 表示"未就绪"（DSL body 内 `ctx.up == 0` 保护即可）。

## 4. 双 rail：`ema_channel_push` / `ema_channel_current`

```python
# 增量推入
new_up_st, new_dw_st = ema_channel_push(
    up_st_sum, up_st_count, up_st_ema,
    dw_st_sum, dw_st_count, dw_st_ema,
    cur_high, cur_low, p
)
# 当前通道值
up, dw = ema_channel_current(
    up_st_sum, up_st_count, up_st_ema,
    dw_st_sum, dw_st_count, dw_st_ema,
    cur_high, cur_low, p
)
```

DSL body 标准用法（参考 `strategies/channel_deviation.py`）：

```python
ctx.up_st_sum, ctx.up_st_count, ctx.up_st_ema, ctx.dw_st_sum, ctx.dw_st_count, ctx.dw_st_ema = ema_channel_push(
    ctx.up_st_sum, ctx.up_st_count, ctx.up_st_ema,
    ctx.dw_st_sum, ctx.dw_st_count, ctx.dw_st_ema,
    ctx.cur_high, ctx.cur_low, ctx.p_N
)
ctx.up, ctx.dw = ema_channel_current(
    ctx.up_st_sum, ctx.up_st_count, ctx.up_st_ema,
    ctx.dw_st_sum, ctx.dw_st_count, ctx.dw_st_ema,
    ctx.cur_high, ctx.cur_low, ctx.p_N
)
```

## 5. 数据就绪时间线（tf1=21 为例）

| 已闭合 bar 数 | push 后状态 | current(pending) 返回 |
|---|---|---|
| 0 ~ 19 | 累加中，无 EMA | inf（count<20） |
| 20（count=20=p-1） | 累加中 | `(sum+pending)/21`，首个试探 EMA |
| 21（count=21=p） | `ema = sum/21`（SMA seed） | `pending*k + ema*(1-k)` |
| 之后每 bar | 正常递推 | 同上 |

因此：**策略最早能在"第 21 个周期 bar"看到通道值**。回测默认预热 365 天，即使 1d 周期也有充足历史。

## 6. 策略层使用：DSL body 两段式

```
=== 指标更新 (每根 bar 调 ema_channel_push + ema_channel_current) ===
ctx.up_st_*, ctx.dw_st_* = ema_channel_push(...)
ctx.up, ctx.dw = ema_channel_current(...)

=== 信号判定 (原有逻辑) ===
if ctx.cur_ts != ctx.lock_ts: ...   # 桶切换清锁
...                                 # 偏离计算 / 锁存 / 触发
```

EMA 状态字段必须在 `state_spec` 中声明（如 `up_st_sum: float` / `up_st_count: int` / `up_st_ema: float`），三端自动单名空间投影。

## 7. 复杂度对比（为什么做增量）

朴素做法：每次回调对全部闭合桶重算 EMA → O(n)/根，n 为桶数，全程 O(n²)。
增量做法：`push` O(1)/桶 + `current` O(1)/根 → 全程 O(总根数)。
单文件回测（百万级 1m bar）与实盘逐根推送都因此可行。

## 8. 顶层 API

```python
from evtrade.indicators import (
    ema,            # 批量版
    ema_channel,    # 批量版
    ema_push, ema_current,           # 增量版（DSL 调用）
    ema_channel_push, ema_channel_current,  # 增量版（DSL 调用）
)
```

旧 `evtrade.core.incremental_indicators.IncrementalEMA` / `EMAChannel` 已删除。