# 05 指标计算：EMA 与通道轨

> 相关源码：`indicators/ema.py`（`evtrade/indicators/ema.py`）。
> 本文档为 2026-09-10 重构版（strategy-step-only）—— 删 `ema_push / ema_current` 双轨标量 API,
> 改用 `ema_step(state, value, p) -> (state, ema)` 单函数 + `@dataclass EMAState`。
> xp 批量版 `xp_ema / xp_ema_channel` 保留供 engine fast-path 调用。

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

`evtrade/indicators/ema.py` 提供**三类形态**，口径一致：

| 形态 | 函数 | 用途 | 复杂度 |
|---|---|---|---|
| 批量 (xp) | `xp_ema(xp, values, p)` | engine fast-path 算全序列 | O(n)，xp 算子 |
| 批量 (xp) | `xp_ema_channel(xp, h, l, p)` | 通道策略批量版 | O(n) |
| step 增量 | `ema_step(EMAState, value, p) -> (EMAState, ema)` | 策略 `step(state, bar)` 调 | O(1) |
| step 增量 | `ema_channel_step(EMAChannelState, h, l, p) -> (EMAChannelState, up, dw)` | 通道策略 step 调 | O(1) |

> **2026-09-10 变化 (strategy-step-only)**：
> - 删旧 `ema_push(s_sum, s_count, s_ema, value, p)` / `ema_current(...)` 6/3 标量 API
> - 新 `ema_step(state, value, p) -> (state, ema)`: state 是 `@dataclass EMAState(sum, count, ema)`, in/out 单 dataclass
> - 公式与浮点路径与旧版逐位一致（同 k=2/(p+1) SMA seed + 递推）
> - 旧 push/current 改作 deprecated shim, 后续 change 删除

## 3. step 版：`ema_step`

state 是 `@dataclass EMAState(sum, count, ema)`, 初值 `EMAState()` 全 0。

```python
@dataclass
class EMAState:
    sum: float = 0.0
    count: int = 0
    ema: float = 0.0

def ema_step(state: EMAState, value: float, p: int) -> tuple[EMAState, float]:
    if state.count < p:
        new_sum = state.sum + value
        new_count = state.count + 1
        if new_count < p:
            return EMAState(new_sum, new_count, 0.0), 0.0
        new_ema = new_sum / p                                # SMA seed
        return EMAState(new_sum, new_count, new_ema), new_ema
    k = 2.0 / (p + 1.0)
    new_ema = value * k + state.ema * (1.0 - k)             # 递推
    return EMAState(state.sum, state.count + 1, new_ema), new_ema
```

桶闭合 / 每根 bar 推入时调用，O(1)：

```
count <  p :  sum += value; count++; 若 count == p: ema = sum/p      ← SMA seed
count >= p :  ema = value*k + ema*(1-k); count++
```

> 与旧版 (`inf` 表示未就绪) 不同的选择: 用 `0.0` 表示未就绪 (state.ema = 0.0);
> 策略代码用 `if up == 0.0 or dw == 0.0: return 0` 守卫即可。
> 数值口径 (count==p-1 → SMA seed) 与旧版完全一致。

## 4. 双 rail：`ema_channel_step`

```python
@dataclass
class EMAChannelState:
    up: EMAState = field(default_factory=EMAState)
    dw: EMAState = field(default_factory=EMAState)

def ema_channel_step(state: EMAChannelState, h: float, l: float, p: int
                    ) -> tuple[EMAChannelState, float, float]:
    new_up, up = ema_step(state.up, h, p)
    new_dw, dw = ema_step(state.dw, l, p)
    return EMAChannelState(up=new_up, dw=new_dw), up, dw
```

策略层典型用法（参考 `evtrade/strategies/channel_deviation.py::step`）：

```python
def step(self, state, bar, params):
    if bar["mark"] == 0:
        return state, 0                          # 预热段

    tf1 = int(params["tf1"])
    cur_ts = int(bar["ts"])
    cur_high = float(bar["h"])
    cur_low = float(bar["l"])

    # 桶切换: 旧桶 high/low 闭锁入 EMA
    if state.has_prev and state.prev_ts != cur_ts:
        state.ema, _, _ = ema_channel_step(state.ema, state.cur_high, state.cur_low, tf1)

    state.prev_ts = cur_ts
    state.cur_high = cur_high; state.cur_low = cur_low
    state.has_prev = True

    # 当前通道值 (含 pending)
    state.ema, up, dw = ema_channel_step(state.ema, cur_high, cur_low, tf1)
    if up == 0.0 or dw == 0.0:
        return state, 0                          # 通道未就绪
    # ... 算偏离 + FSM
    return state, sig
```

## 5. 数据就绪时间线（tf1=21 为例）

| 已 step 调用次数 | EMAState.count | ema_step 返回的 ema |
|---|---|---|
| 0 ~ 20 | 累加中 (count < p) | 0.0 |
| 21 (count=21=p) | `ema = sum/21`（SMA seed） | `sum/21` |
| 之后每 step | 正常递推 | `value*k + ema*(1-k)` |

因此：**策略最早能在"第 21 次 step"看到通道值**。回测默认预热 365 天, 即使 1d 周期也有充足历史。

## 6. 策略层使用：step 内混合写法

策略 `step(state, bar, params)` body 内:

```python
# 1) EMA 通道 (step 增量; state 由 engine 持有)
state.ema, up, dw = ema_channel_step(state.ema, bar["h"], bar["l"], tf1)

# 2) NaN -> 0 (引擎 batched 路径下, _compute_devs_xp 内部安全 mask; step 路径无需)
low_dev   = (dw - bar["l"]) / dw * 100.0      # 通道未就绪时 dw==0 -> dev==0, FSM 不触发
high_dev  = (bar["h"] - up) / up * 100.0
```

batched 路径（`VectorizedEngine._compute_signals`）也调同一份 step, 引擎循环维护 state。

## 7. 复杂度对比

朴素做法：每次回调对全部闭合桶重算 EMA → O(n)/根, n 为桶数, 全程 O(n²)。
step 增量：`ema_step` O(1)/次 + state 跨调用持续 → 全程 O(总调用数)。
单文件回测（百万级 1m bar）与实盘逐根推送都因此可行。
**批量路径**用 `xp_ema_channel` 一次性算全序列（O(n) 总）, GPU 路径下走 cuBLAS 预编译 kernel。

## 8. 顶层 API

```python
from evtrade.indicators import (
    # 批量 xp 版 (engine fast-path)
    xp_ema, xp_ema_channel,

    # step 增量版 (strategy-step-only, 2026-09-10)
    EMAState, EMAChannelState,
    ema_step, ema_channel_step,

    # Deprecated shim (过渡期; 后续删除)
    ema_push, ema_current, ema_channel_push, ema_channel_current,

    # 同形态还有 xp_atr / xp_rsi / xp_bollinger / atr_step / rsi_step / sma_step / boll_step
)
```

旧 `evtrade.core.incremental_indicators.IncrementalEMA` / `EMAChannel` 已删除（旧 `unify-strategy-contract`）。