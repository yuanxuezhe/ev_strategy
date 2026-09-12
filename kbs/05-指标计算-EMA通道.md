# 05 指标计算：EMA 与通道轨

> **2026-09-13 重要更新**: framework 不再持有资金/持仓/撮合/PnL/收益概念; 策略 `step` 内部自管。`Account` / `Executor` / `SimulatedExecutor` / `trade_decision` / `metrics.summarize` 已下线; `core/metrics` / `core/replay` / `core/permutation` / `core/config` 已删; `replay` 子命令 + `--against-ref` 已下线; `--init-cash --buy-pct --sell-pct --all-in --trade-qty --warmup-days --data-cache` CLI flag 已删 (走 `--params`).

> 相关源码：`indicators/ema.py`（`evtrade/indicators/ema.py`）。
> 本文档为 2026-09-10 重构版（strategy-step-only）—— 删旧双轨标量 API,
> 改用 `ema_step(state, value, p) -> (state, ema)` 单函数 + `@dataclass EMAState`；
> 旧 xp 批量版已删除 —— 引擎 vectorized 路径 = 逐桶调同一份
> 策略 `step`，无独立批量 EMA 算子。

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
| step 增量 | `ema_step(EMAState, value, p) -> (EMAState, ema)` | 策略 `step(state, bar)` 调 | O(1) |
| step 增量 | `ema_channel_step(EMAChannelState, h, l, p) -> (EMAChannelState, up, dw)` | 通道策略 step 调 | O(1) |
| numpy 批量参考 | `ema(values, p)` | ndarray 输入输出，reconcile 参考 / 测试用 | O(n) |
| numpy 批量参考 | `ema_channel(highs, lows, p)` | 上轨 + 下轨，reconcile 参考 / 测试用 | O(n) |
| torch 批量 | `torch_ema(values: Tensor[T], p) -> Tensor[T]` | GPU-batched sweep `batched_step` hook 专用 opt-in 形态；float64 与 numpy `ema` bitwise 一致 | O(n) |

> **2026-09-11 变化 (gpu-batched-sweep)**：
> - 新增 `torch_ema(values, p) -> Tensor[T]` 批量形态，仅供 `batched_step` hook 内部按 period 分组调用。
> - 算法与 numpy 参考版同式（`k = 2/(p+1)`，前 `p-1` 个返回 `0.0` 与 `ema_step` 对齐而非 numpy 的 NaN）；
>   seed 必须用 `numpy.sum` 取首段避免 GPU 串行 cumsum 与 numpy pairwise 求和的 1-bit 末位差。
> - 不暴露为策略 `step` 的替代入口；`step` 仍走 `ema_step` / `ema_channel_step`。

> **2026-09-10 变化 (strategy-step-only)**：
> - 删旧双轨标量 API（6/3 元组形式）
> - 新 `ema_step(state, value, p) -> (state, ema)`: state 是 `@dataclass EMAState(sum, count, ema)`, in/out 单 dataclass
> - 公式与浮点路径与旧版逐位一致（同 k=2/(p+1) SMA seed + 递推）
> - 旧双轨 API 已删除（deprecated shim 一并清除）
> - 旧 xp 批量版 (engine fast-path) 已删除: 引擎 vectorized 路径
>   (`VectorizedEngine._compute_signals`) 逐桶调同一份策略 `step`，无独立批量 EMA 算子

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
> 注意 numpy 批量参考版 `ema(values, p)` 用 `NaN` 表示未就绪（前 p-1 个），仅用于
> 离线 reconcile 对比，不参与回测路径。

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

# 2) 偏离 (stateless 算)
low_dev   = (dw - bar["l"]) / dw * 100.0      # 通道未就绪时 dw==0 -> dev==0, FSM 不触发
high_dev  = (bar["h"] - up) / up * 100.0
```

批量(vectorized)路径（`VectorizedEngine._compute_signals`）= 引擎在循环里**逐桶调同一份 `step`**，
引擎维护 state；**无独立批量 EMA 算子**（旧 xp 批量快路径已删除）。

## 7. 复杂度对比

朴素做法：每次回调对全部闭合桶重算 EMA → O(n)/根, n 为桶数, 全程 O(n²)。
step 增量：`ema_step` O(1)/次 + state 跨调用持续 → 全程 O(总调用数)。
单文件回测（百万级 1m bar）与实盘逐根推送都因此可行。
**批量(vectorized)路径** = 引擎在 `_compute_signals` 里逐桶调同一份 `step`（无独立批量 EMA 算子），
总复杂度仍是 O(总桶数)，CPU/GPU 共用同一份策略代码。

## 8. 顶层 API

```python
from evtrade.indicators import (
    # step 增量版 (strategy-step-only, 策略 step 内调)
    EMAState, EMAChannelState,
    ema_step, ema_channel_step,

    # numpy 批量参考版 (reconcile 参考实现 / 测试用, 非引擎路径)
    ema, ema_channel,

    # torch 批量版 (batched_step hook 内部用, opt-in; 2026-09-11 新增)
    torch_ema,
)
```

旧 `evtrade.core.incremental_indicators.IncrementalEMA` / `EMAChannel` 已删除（旧 `unify-strategy-contract`）。