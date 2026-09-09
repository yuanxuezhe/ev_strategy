# 05 指标计算：EMA 与通道轨

> 相关源码：`indicators/ema.py`（`evtrade/indicators/ema.py`）。
> 历史包袱：`evtrade/core/incremental_indicators.py` 已在 change `2026-09-09-decouple-indicators-from-framework` 中删除——framework 不再持有 EMA 状态，所有指标计算由策略通过 `evtrade/indicators/` 在 `compute_signals` 内部自维护。
> 本文档为 2026-09-09 统一 CPU/GPU 重构版（删 `numba`、删 `CUDA_DEVICE_*`）。

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

`evtrade/indicators/ema.py` 提供**双形态**，口径一致：

| 形态 | 函数 | 用途 | 复杂度 |
|---|---|---|---|
| 批量 (xp) | `xp_ema(xp, values, p)` | `compute_signals` 主体 (向量化引擎 / sweep) | O(n)，xp 算子 |
| 批量 (xp) | `xp_ema_channel(xp, h, l, p)` | 通道策略批量版 | O(n) |
| 增量 (Python 标量) | `ema_push / ema_current` | `compute_signals_for_one_bar` 逐 bar O(1) | O(1) |
| 增量 (Python 标量) | `ema_channel_push / ema_channel_current` | 通道逐 bar | O(1) |

> **2026-09-09 变化**：
> - 旧版 `ema / ema_channel` 批量函数（return `np.ndarray`）保留，新版 `xp_ema / xp_ema_channel` 接受 `xp` 第一参数（numpy/cupy 兼容）。
> - 旧版 `@njit` 增量函数重写为**纯 Python**（保留 3-6 标量 in/out API）。无 `numba` import、无 `CUDA_DEVICE_*` 字符串常量。
> - 表达式与浮点路径与原 numba 版逐位一致 (同一份 k=2/(p+1) SMA seed + 递推)。

## 3. 增量版：`ema_push` / `ema_current`

增量 state 形状为 `(sum, count, ema)` 三元组，初始 `(0.0, 0, 0.0)`。
`compute_signals_for_one_bar` 在策略 instance 维护 (`self._up_st = (0.0, 0, 0.0)`)。

### `ema_push(s_sum, s_count, s_ema, value, p) -> (sum, count, ema)`

桶闭合 / 每根 bar 推入时调用，O(1)：

```
count <  p :  sum += value; count++; 若 count == p: ema = sum/p      ← SMA seed
count >= p :  ema = value*k + ema*(1-k); count++
```

返回新 state（按值，Python tuple）。

### `ema_current(s_sum, s_count, s_ema, pending, p) -> float`

带当前未闭合 bar 的"试探性 EMA"，O(1)，不改 state：

```
count <  p-1 :  返回 0.0（数据不足；策略代码用 up==0 守卫即可）
count == p-1 :  返回 (sum + pending) / p        ← pending 凑满 p 个，得 SMA seed
count >= p   :  返回 pending*k + ema*(1-k)
```

> 与旧 numba 版 (`inf` 表示未就绪) 不同的选择: 用 `0.0` 表示未就绪；
> 策略代码用 `if up == 0.0 or dw == 0.0: return 0` 守卫即可。
> 数值口径 (count==p-1 → SMA seed) 与旧版完全一致。

## 4. 双 rail：`ema_channel_push` / `ema_channel_current`

```python
# 增量推入 (6 标量 in → 6 标量 out)
new_up_st, new_dw_st = ema_channel_push(
    up_st_sum, up_st_count, up_st_ema,
    dw_st_sum, dw_st_count, dw_st_ema,
    cur_high, cur_low, p
)
# 当前通道值 (6 标量 in → 2 标量 out)
up, dw = ema_channel_current(
    up_st_sum, up_st_count, up_st_ema,
    dw_st_sum, dw_st_count, dw_st_ema,
    cur_high, cur_low, p
)
```

策略层典型用法（参考 `evtrade/strategies/channel_deviation.py::compute_signals_for_one_bar`）：

```python
def compute_signals_for_one_bar(self, xp, bar, params):
    tf1 = int(params["tf1"])
    cur_ts = int(bar["ts"])
    cur_high = float(bar["h"])
    cur_low = float(bar["l"])

    # 桶切换: 旧桶 high/low 闭锁入 EMA
    if self._has_prev and self._prev_ts != cur_ts:
        us, uc, ue = self._up_st
        us, uc, ue = ema_push(us, uc, ue, self._cur_high, tf1)
        self._up_st = (us, uc, ue)
        ds, dc, de = self._dw_st
        ds, dc, de = ema_push(ds, dc, de, self._cur_low, tf1)
        self._dw_st = (ds, dc, de)

    self._prev_ts = cur_ts; self._cur_high = cur_high; self._cur_low = cur_low
    self._has_prev = True

    # 当前通道值 (含 pending)
    us, uc, ue = self._up_st; ds, dc, de = self._dw_st
    up = ema_current(us, uc, ue, cur_high, tf1)
    dw = ema_current(ds, dc, de, cur_low, tf1)
    if up == 0.0 or dw == 0.0:
        return 0
    ...
```

## 5. 数据就绪时间线（tf1=21 为例）

| 已闭合 bar 数 | push 后状态 | current(pending) 返回 |
|---|---|---|
| 0 ~ 19 | 累加中，无 EMA | 0.0（count<20） |
| 20（count=20=p-1） | 累加中 | `(sum+pending)/21`，首个试探 EMA |
| 21（count=21=p） | `ema = sum/21`（SMA seed） | `pending*k + ema*(1-k)` |
| 之后每 bar | 正常递推 | 同上 |

因此：**策略最早能在"第 21 个周期 bar"看到通道值**。回测默认预热 365 天，即使 1d 周期也有充足历史。

## 6. 策略层使用：compute_signals 内混合写法

批量路径（vectorized 引擎 / sweep）:

```python
def compute_signals(self, xp, bars, params):
    tf1 = int(params["tf1"])
    up, dw = xp_ema_channel(xp, bars["h"], bars["l"], tf1)
    # NaN -> 0 (未就绪时偏离不触发)
    up_v = xp.where(xp.isnan(up), 0.0, up)
    dw_v = xp.where(xp.isnan(dw), 0.0, dw)
    low_dev = (dw_v - bars["l"]) / dw_v * 100.0
    high_dev = (bars["h"] - up_v) / up_v * 100.0
    ...
```

逐 bar 路径（Engine.on_bars / 实盘）见 §4。

## 7. 复杂度对比（为什么做增量）

朴素做法：每次回调对全部闭合桶重算 EMA → O(n)/根，n 为桶数，全程 O(n²)。
增量做法：`push` O(1)/桶 + `current` O(1)/根 → 全程 O(总根数)。
单文件回测（百万级 1m bar）与实盘逐根推送都因此可行。
**批量路径**用 `xp_ema_channel` 一次性算全序列（O(n) 总），GPU 路径下走 cuBLAS 预编译 kernel。

## 8. 顶层 API

```python
from evtrade.indicators import (
    xp_ema,             # 批量 xp 版
    xp_ema_channel,     # 批量 xp 版
    ema_push, ema_current,                          # 增量 Python
    ema_channel_push, ema_channel_current,          # 增量 Python
    # 同形态还有 xp_atr / xp_rsi / xp_bollinger / atp_push / rsi_push / boll_push
)
```

旧 `evtrade.core.incremental_indicators.IncrementalEMA` / `EMAChannel` 已删除；
旧 `ema(values)` / `ema_channel(values)` 批量函数被 `xp_ema(xp, ...)` / `xp_ema_channel(xp, ...)` 取代（多一个 `xp` 第一参数）。
