from __future__ import annotations
"""EMA / EMAChannel 指标

两种形态:
  1. step 增量版 (ema_step, ema_channel_step): 策略 step() 逐桶调用, state 用 dataclass
  2. numpy 批量参考版 (ema, ema_channel): ndarray 输入输出 (reconcile 参考 / 测试用)

EMA 公式:
  - count < p:   累加 sum; count += 1; ema 仍 NaN (批量版) / 0 (step 版)
  - count == p:  ema = sum / p  (SMA seed)
  - count >= p:  ema = value*k + ema*(1-k), k = 2/(p+1)
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EMAState:
    """EMA 增量 state; ema_step in/out"""
    sum: float = 0.0
    count: int = 0
    ema: float = 0.0


@dataclass
class EMAChannelState:
    """EMA 通道增量 state"""
    up: EMAState = field(default_factory=EMAState)
    dw: EMAState = field(default_factory=EMAState)


def ema_step(state: EMAState, value: float, p: int) -> tuple:
    """EMA 单步: state + 1 标量 -> (new_state, ema_value)"""
    if state.count < p:
        new_sum = state.sum + value
        new_count = state.count + 1
        if new_count < p:
            return EMAState(sum=new_sum, count=new_count, ema=0.0), 0.0
        new_ema = new_sum / p
        return EMAState(sum=new_sum, count=new_count, ema=new_ema), new_ema
    k = 2.0 / (p + 1.0)
    new_ema = value * k + state.ema * (1.0 - k)
    return EMAState(sum=state.sum, count=state.count + 1, ema=new_ema), new_ema


def ema_channel_step(state: EMAChannelState, h: float, l: float,
                     p: int) -> tuple:
    new_up, up = ema_step(state.up, h, p)
    new_dw, dw = ema_step(state.dw, l, p)
    return EMAChannelState(up=new_up, dw=new_dw), up, dw


def ema(values, p: int):
    """EMA(values, p): numpy ndarray 输入返 ndarray。前 p-1 个 NaN。"""
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
    return ema(highs, p), ema(lows, p)
