# 03 指标:EMA 通道

> 单指标 EMA, 三种形态(`ema_step` 增量 / `ema` numpy 批量 / `torch_ema` torch 批量)
>
> framework 仅暴露 EMA; 其它指标(`atr / boll / rsi` 等)已在 2026-09-09 重构中删除。
> 所有指标的 `*_step` 增量版是策略 `step` body 的内部维护入口(`@dataclass` state,
> engine 跨调用持有, 策略无 instance attr)。
>
> 相关源码: `evtrade/indicators/ema.py`

## 1. 通道轨定义(通达信蓝色通道轨)

```
UP := EMA(H, tf1)    上轨(对周期桶最高价做 EMA)
DW := EMA(L, tf1)    下轨(对周期桶最低价做 EMA)
tf1 = 21(默认, 由策略 params_spec 自定义)
```

上轨是"最高价的 tf1 周期 EMA", 下轨是"最低价的 tf1 周期 EMA",
构成一条随价格平滑移动的通道。详细偏离量公式见 [04-策略 §偏离](04-策略.md)。

## 2. EMA 数学定义

- 平滑系数: `k = 2/(p+1)`(p=21 时 k ≈ 0.0909)
- **种子(seed)**: 前 p 个值做简单平均 `SMA = sum(前p个)/p`, 作为第一个 EMA
- 之后递推: `EMA_t = price_t × k + EMA_{t-1} × (1-k)`

`evtrade/indicators/ema.py` 提供**三类形态**, 口径一致:

| 形态 | 函数 | 用途 | 复杂度 |
|---|---|---|---|
| step 增量 | `ema_step(EMAState, value, p) -> (EMAState, ema)` | 策略 `step` 内调 | O(1) |
| step 增量 | `ema_channel_step(EMAChannelState, h, l, p) -> (EMAChannelState, up, dw)` | 通道策略 step 调 | O(1) |
| numpy 批量参考 | `ema(values, p)` | ndarray 输入输出, reconcile / 测试用 | O(n) |
| numpy 批量参考 | `ema_channel(highs, lows, p)` | 上轨 + 下轨, reconcile / 测试用 | O(n) |
| torch 批量 | `torch_ema(values: Tensor[T], p) -> Tensor[T]` | GPU-batched sweep `batched_step` hook 专用 | O(n) |

## 3. step 增量版

### 3.1 `ema_step` —— 单值 EMA

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

state 初值 `EMAState()` 全 0。每根 bar / 桶闭合时调一次, O(1):

```
count <  p :  sum += value; count++; 若 count == p: ema = sum/p      ← SMA seed
count >= p :  ema = value*k + ema*(1-k); count++
```

> 与旧版(`inf` 表示未就绪)不同的选择: 用 `0.0` 表示未就绪(state.ema = 0.0);
> 策略代码用 `if up == 0.0 or dw == 0.0: return 0` 守卫即可。
> numpy 批量参考版 `ema(values, p)` 用 `NaN` 表示未就绪(前 p-1 个), 仅用于离线 reconcile,
> 不参与回测路径。

### 3.2 `ema_channel_step` —— 双 rail(上轨 + 下轨)

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

策略层典型用法(`evtrade/strategies/channel_deviation.py`):

```python
def step(self, state, bar, params):
    if bar["mark"] == 0:
        return state, 0                          # 预热段

    tf1 = int(params["tf1"])

    # 桶切换: 旧桶 high/low 闭锁入 EMA
    if state.has_prev and state.prev_ts != int(bar["ts"]):
        state.ema, _, _ = ema_channel_step(state.ema, state.cur_high, state.cur_low, tf1)

    state.prev_ts = int(bar["ts"])
    state.cur_high = float(bar["h"]); state.cur_low = float(bar["l"])
    state.has_prev = True

    # 当前通道值 (含 pending)
    state.ema, up, dw = ema_channel_step(state.ema, bar["h"], bar["l"], tf1)
    if up == 0.0 or dw == 0.0:
        return state, 0                          # 通道未就绪
    # ... 算偏离 + FSM
    return state, sig
```

## 4. 数据就绪时间线(tf1=21 为例)

| 已 step 调用次数 | EMAState.count | ema_step 返回的 ema |
|---|---|---|
| 0 ~ 20 | 累加中(count < p) | 0.0 |
| 21(count=21=p) | `ema = sum/21`(SMA seed) | `sum/21` |
| 之后每 step | 正常递推 | `value*k + ema*(1-k)` |

策略最早能在"第 21 次 step"看到通道值。
回测默认预热 365 天, 即使 1d 周期也有充足历史。

## 5. torch 批量版(`torch_ema`)

`torch_ema(values: Tensor[T], p: int) -> Tensor[T]` 是 GPU-batched sweep 的 opt-in 形态
(`batched_step` hook 内部按 period 分组调用):

- 输入 `[T]`、输出 `[T]`、`dtype/device` 保留(自动转 float64)
- 前 `p-1` 个返回 `0.0`(跟 `ema_step` 对齐而非 numpy 的 NaN)
- 算法: `cumsum` seed + 递推尾段 `out[i] = values[i]*k + out[i-1]*(1-k)`, `k = 2/(p+1)`
- **跟 numpy `ema()` 参考版 float64 bit-equal**(`np.array_equal` 通过)
- seed 必须用 `numpy.sum` 取首段避免 GPU 串行 cumsum 与 numpy pairwise 求和的 1-bit 末位差

不暴露为策略 `step` 的替代入口; `step` 仍走 `ema_step` / `ema_channel_step`。

## 6. 顶层 API

```python
from evtrade.indicators import (
    # step 增量版 (策略 step 内调)
    EMAState, EMAChannelState,
    ema_step, ema_channel_step,

    # numpy 批量参考版 (reconcile 参考 / 测试用, 非引擎路径)
    ema, ema_channel,

    # torch 批量版 (batched_step hook 内部用, opt-in)
    torch_ema,
)
```

旧 `evtrade.core.incremental_indicators.IncrementalEMA` / `EMAChannel` 已删除;
旧 `xp_ema` / `xp_ema_channel` / `xp_ema_torch` 批量 xp 算子随 cupy 一并删除;
旧 `ema_push` / `ema_current` 已统一为 `ema_step`。
