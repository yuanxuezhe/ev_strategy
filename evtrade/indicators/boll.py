from __future__ import annotations
"""Bollinger Bands (布林带) 指标 (纯 ndarray 版 + 纯 Python 增量版)

================================================================
✅  可改层 (indicators 子包)  ✅
================================================================
中轨 = MA(N) (简单移动平均)
上轨 = 中轨 + k * Std(N)
下轨 = 中轨 - k * Std(N)
默认 N=20, k=2 (与 TradingView 一致)。

@Python 增量版 (Engine 路径调用):
  - sma_push(state, value) -> state
  - sma_current(state, pending, p) -> float
  - boll_push(state, value) -> state
  - boll_current(state, pending, p, k) -> (mid, upper, lower)
简化: 增量版用 running mean (sum/count), 不严格 sliding window SMA (避免
维护 buffer)。批量校核请用 sma(values, p) 纯函数版。
state:
  sma:  (sum: float, count: int)
  boll: (sum: float, sumsq: float, count: int)
"""
import numpy as np


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


# ============ 纯 Python 增量版 (Engine.on_bars 路径) ============

def sma_push(state, value):
    """SMA running-mean 增量推入: state=(sum, count)"""
    s_sum = state[0]
    count = state[1]
    return s_sum + value, count + 1


def sma_current(state, pending, p):
    """SMA 当前值 (含 pending, 不改 state); 数据不足返回 0.0

    表达式:
      count == 0: 0.0
      count < p-1: 0.0
      count == p-1: (sum + pending) / p  (凑齐 p 个)
      count >= p: (sum + pending) / (count + 1)  (running mean)
    """
    s_sum = state[0]
    count = state[1]
    if count < p - 1:
        return 0.0
    if count == p - 1:
        return (s_sum + pending) / p
    return (s_sum + pending) / (count + 1)


def boll_push(state, value):
    """Bollinger 增量推入: state=(sum, sumsq, count)"""
    s_sum = state[0]
    s_sumsq = state[1]
    count = state[2]
    return s_sum + value, s_sumsq + value * value, count + 1


def boll_current(state, pending, p, k):
    """Bollinger 当前值 (含 pending, 不改 state); 返回 (mid, upper, lower)

    数据不足时 (count < p-1) 返回 (0.0, 0.0, 0.0)。
    """
    s_sum = state[0]
    s_sumsq = state[1]
    count = state[2]
    if count < p - 1:
        return 0.0, 0.0, 0.0
    new_sum = s_sum + pending
    new_sumsq = s_sumsq + pending * pending
    new_count = count + 1
    mid = new_sum / new_count
    var = new_sumsq / new_count - mid * mid
    if var < 0.0:
        var = 0.0   # 数值误差保护
    std = var ** 0.5
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower
