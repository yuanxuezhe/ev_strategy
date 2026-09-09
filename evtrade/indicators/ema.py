from __future__ import annotations
"""EMA / EMAChannel 指标

================================================================
✅  可改层 (indicators 子包)  ✅
================================================================
三种形态:
  1. 纯 xp 版 (xp_ema, xp_ema_channel): 一次性批量算整段序列,
     接受 xp (numpy|cupy) 模块; engine fast-path 调。
  2. step 增量版 (ema_step, ema_channel_step): 策略 step() 调;
     state 用 @dataclass EMAState / EMAChannelState 承载, 返回 (new_state, ema)。
  3. 纯函数版 (ema, ema_channel): numpy 返 ndarray, jupyter / 复盘用。

@njit / CUDA __device__ 渲染器已下线 (DSL 整体废弃); 本文件不依赖 numba。

@step API (策略 step 调):
  - ema_step(state: EMAState, value: float, p: int) -> (EMAState, float)
  - ema_channel_step(state: EMAChannelState, h: float, l: float, p: int)
      -> (EMAChannelState, float, float)

EMA 公式 (与批量版逐位一致):
  - count < p:   累加 sum; count += 1; ema 仍 0
  - count == p:  ema = sum / p  (SMA seed)
  - count >= p:  ema = value*k + ema*(1-k), k = 2/(p+1)
"""
from dataclasses import dataclass, field

import numpy as np


# ============ step state (dataclass) ============

@dataclass
class EMAState:
    """EMA 增量 state; ema_step in/out"""
    sum: float = 0.0
    count: int = 0
    ema: float = 0.0


@dataclass
class EMAChannelState:
    """EMA 通道增量 state; ema_channel_step in/out"""
    up: EMAState = field(default_factory=EMAState)
    dw: EMAState = field(default_factory=EMAState)


# ============ 纯 xp 版 (engine fast-path 调用) ============

def xp_ema(xp, values, p: int):
    """EMA 批量版 (xp 兼容); 前 p-1 根 NaN, 之后递推。

    SMA seed = 前 p 个均值; EMA_t = value*k + EMA_{t-1}*(1-k), k=2/(p+1)。
    """
    n = len(values)
    out = xp.full(n, xp.nan, dtype=xp.float64)
    if n < p:
        return out
    k = 2.0 / (p + 1.0)
    seed = xp.sum(values[:p]) / p
    out[p - 1] = seed
    e = float(seed)
    for i in range(p, n):
        e = float(values[i]) * k + e * (1.0 - k)
        out[i] = e
    return out


def xp_ema_channel(xp, highs, lows, p: int):
    """EMA 通道: 上轨 = EMA(H, p), 下轨 = EMA(L, p)"""
    return xp_ema(xp, highs, p), xp_ema(xp, lows, p)


# ============ step 增量版 (策略 step 调用, strategy-step-only) ============

def ema_step(state: EMAState, value: float, p: int) -> tuple[EMAState, float]:
    """EMA 单步: state + 1 标量 -> (new_state, ema_value)

    规则 (与原 ema_push + ema_current 等价):
      - count < p:   sum += value; count += 1; ema 仍 0
      - count == p:  ema = sum / p (SMA seed, 凑齐 p 个)
      - count >= p:  ema = value*k + ema*(1-k), k = 2/(p+1)
    """
    if state.count < p:
        new_sum = state.sum + value
        new_count = state.count + 1
        if new_count < p:
            return EMAState(sum=new_sum, count=new_count, ema=0.0), 0.0
        new_ema = new_sum / p  # SMA seed
        return EMAState(sum=new_sum, count=new_count, ema=new_ema), new_ema
    # count >= p: EMA 递推
    k = 2.0 / (p + 1.0)
    new_ema = value * k + state.ema * (1.0 - k)
    return EMAState(sum=state.sum, count=state.count + 1, ema=new_ema), new_ema


def ema_channel_step(state: EMAChannelState, h: float, l: float,
                     p: int) -> tuple[EMAChannelState, float, float]:
    """EMA 通道单步: state + (h, l) -> (new_state, up, dw)

    内部调两次 ema_step, 分别推上轨 (h) 和下轨 (l)。
    """
    new_up, up = ema_step(state.up, h, p)
    new_dw, dw = ema_step(state.dw, l, p)
    return EMAChannelState(up=new_up, dw=new_dw), up, dw


# ============ 纯函数版 (jupyter / 复盘, 返 ndarray) ============

def ema(values, p: int):
    """EMA(values, p): 输入长度 >= p 时返回完整 ndarray (前 p-1 个为 NaN, 之后为递推值)

    SMA seed = 前 p 个均值; 之后 EMA_t = value * k + EMA_{t-1} * (1-k), k = 2 / (p+1)

    返回长度 == len(values); len(values) < p 时全部为 NaN。
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < p:
        return out
    k = 2.0 / (p + 1.0)
    seed = np.sum(values[:p]) / p
    out[p - 1] = seed
    e = float(seed)
    for i in range(p, n):
        e = float(values[i]) * k + e * (1.0 - k)
        out[i] = e
    return out


def ema_channel(highs, lows, p: int):
    """EMA 通道 (numpy): 上轨 = EMA(H, p), 下轨 = EMA(L, p)"""
    return ema(highs, p), ema(lows, p)


# ============ Deprecated shim (2026-09-10: 合并到 *_step, 后续删除) ============

def ema_push(s_sum, s_count, s_ema, value, p):
    """DEPRECATED: 用 ema_step(state, value, p) -> (state, ema).
    此函数仅作过渡期 shim, 后续 change 删除。"""
    state = EMAState(sum=s_sum, count=s_count, ema=s_ema)
    new_state, _ = ema_step(state, value, p)
    return new_state.sum, new_state.count, new_state.ema


def ema_current(s_sum, s_count, s_ema, pending, p):
    """DEPRECATED: 用 ema_step(state, pending, p) -> (state, ema)."""
    state = EMAState(sum=s_sum, count=s_count, ema=s_ema)
    _, ema = ema_step(state, pending, p)
    return ema


def ema_channel_push(us, uc, ue, ds, dc, de, h, l, p):
    """DEPRECATED: 用 ema_channel_step(state, h, l, p)."""
    state = EMAChannelState(
        up=EMAState(sum=us, count=uc, ema=ue),
        dw=EMAState(sum=ds, count=dc, ema=de),
    )
    new_state, _, _ = ema_channel_step(state, h, l, p)
    return (new_state.up.sum, new_state.up.count, new_state.up.ema,
            new_state.dw.sum, new_state.dw.count, new_state.dw.ema)


def ema_channel_current(us, uc, ue, ds, dc, de, h, l, p):
    """DEPRECATED: 用 ema_channel_step(state, h, l, p)."""
    state = EMAChannelState(
        up=EMAState(sum=us, count=uc, ema=ue),
        dw=EMAState(sum=ds, count=dc, ema=de),
    )
    _, up, dw = ema_channel_step(state, h, l, p)
    return up, dw