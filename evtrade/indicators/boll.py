from __future__ import annotations
"""Bollinger Bands (布林带) 指标

================================================================
✅  可改层 (indicators 子包)  ✅
================================================================
中轨 = MA(N) (简单移动平均)
上轨 = 中轨 + k * Std(N)
下轨 = 中轨 - k * Std(N)
默认 N=20, k=2 (与 TradingView 一致)。

@step API (策略 step 调):
  - sma_step(state: SMAState, value, p) -> (SMAState, sma)
  - boll_step(state: BollState, value, p, k) -> (BollState, mid, upper, lower)
简化: 增量版用 running mean (sum/count), 不严格 sliding window SMA (避免
维护 buffer)。批量校核请用 sma(values, p) 纯函数版。
state:
  sma:  SMAState(sum, count)
  boll: BollState(sum, sumsq, count)
"""
from dataclasses import dataclass

import numpy as np


# ============ step state (dataclass) ============

@dataclass
class SMAState:
    """SMA 增量 state (running mean); sma_step in/out"""
    sum: float = 0.0
    count: int = 0


@dataclass
class BollState:
    """Bollinger 增量 state (running sum + sumsq); boll_step in/out"""
    sum: float = 0.0
    sumsq: float = 0.0
    count: int = 0


# ============ 纯函数版 (复盘 / jupyter 友好, 返 ndarray) ============

def sma(values, p: int):
    """SMA(p); 长度不足时返回 NaN ndarray"""
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < p:
        return out
    s = float(np.sum(values[:p]))
    out[p - 1] = s / p
    for i in range(p, n):
        s += float(values[i]) - float(values[i - p])
        out[i] = s / p
    return out


def bollinger(closes, p: int = 20, k: float = 2.0):
    """返回 (middle, upper, lower) 三条 ndarray"""
    n = len(closes)
    middle = sma(closes, p)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    if n < p:
        return middle, upper, lower
    for i in range(p - 1, n):
        m = middle[i]
        var = sum((float(closes[i - j]) - m) ** 2 for j in range(p)) / p
        std = var ** 0.5
        upper[i] = m + k * std
        lower[i] = m - k * std
    return middle, upper, lower


# ============ step 增量版 (策略 step 调用, strategy-step-only) ============

def sma_step(state: SMAState, value: float, p: int) -> tuple[SMAState, float]:
    """SMA running-mean 单步: state + value -> (new_state, sma)

    规则 (数据不足返 0.0):
      - count < p-1: sum += value; count += 1; ret=0
      - count == p-1: sum += value; count += 1; ret=sum/p  (凑齐 p 个 -> SMA seed)
      - count >= p: sum += value; count += 1; ret=sum/count  (running mean)
    """
    new_sum = state.sum + value
    new_count = state.count + 1
    if new_count < p:
        return SMAState(sum=new_sum, count=new_count), 0.0
    if new_count == p:
        return SMAState(sum=new_sum, count=new_count), new_sum / p
    return SMAState(sum=new_sum, count=new_count), new_sum / new_count


def boll_step(state: BollState, value: float, p: int,
              k: float) -> tuple[BollState, float, float, float]:
    """Bollinger 单步: state + value -> (new_state, mid, upper, lower)

    规则 (running sum + sumsq; 数据不足返 3 个 0.0):
      - count < p-1: sum += value; sumsq += value^2; count += 1; ret=0,0,0
      - count >= p-1: mid = sum/count; var = sumsq/count - mid^2; std = sqrt(var); ret 三轨
    """
    new_sum = state.sum + value
    new_sumsq = state.sumsq + value * value
    new_count = state.count + 1
    if new_count < p:
        return BollState(sum=new_sum, sumsq=new_sumsq, count=new_count), 0.0, 0.0, 0.0
    mid = new_sum / new_count
    var = new_sumsq / new_count - mid * mid
    if var < 0.0:
        var = 0.0  # 数值误差保护
    std = var ** 0.5
    upper = mid + k * std
    lower = mid - k * std
    return BollState(sum=new_sum, sumsq=new_sumsq, count=new_count), mid, upper, lower


# ============ Deprecated shim (2026-09-10: 合并到 *_step, 后续删除) ============

def sma_push(state, value):
    """DEPRECATED: 用 sma_step(state, value, p) -> (state, sma).

    原 sma_push 不返回 sma, 只累积 state; 这里保留等价语义 (push 不读)。"""
    return state[0] + value, state[1] + 1


def sma_current(state, pending, p):
    """DEPRECATED: 用 sma_step(state, pending, p) -> (state, sma)."""
    s = SMAState(sum=state[0], count=state[1])
    _, sma_v = sma_step(s, pending, p)
    return sma_v


def boll_push(state, value):
    """DEPRECATED: 用 boll_step(state, value, p, k) -> (state, mid, upper, lower).

    原 boll_push 不返回指标, 只累积 state; 这里保留等价语义。"""
    return state[0] + value, state[1] + value * value, state[2] + 1


def boll_current(state, pending, p, k):
    """DEPRECATED: 用 boll_step(state, pending, p, k) -> (state, mid, upper, lower)."""
    s = BollState(sum=state[0], sumsq=state[1], count=state[2])
    _, mid, upper, lower = boll_step(s, pending, p, k)
    return mid, upper, lower