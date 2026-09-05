from __future__ import annotations
"""Bollinger Bands (布林带) 纯函数

================================================================
✅  可改层 (indicators 子包)  ✅
================================================================
中轨 = MA(N) (简单移动平均)
上轨 = 中轨 + k * Std(N)
下轨 = 中轨 - k * Std(N)
默认 N=20, k=2 (与 TradingView 一致)。
"""
from typing import Sequence


def sma(values: Sequence[float], p: int) -> list[float | None]:
    """SMA(p); 长度不足时返回 None"""
    n = len(values)
    out: list[float | None] = [None] * n
    if n < p:
        return out
    s = sum(values[:p])
    out[p - 1] = s / p
    for i in range(p, n):
        s += values[i] - values[i - p]
        out[i] = s / p
    return out


def bollinger(closes: Sequence[float], p: int = 20, k: float = 2.0):
    """返回 (middle, upper, lower) 三条线"""
    n = len(closes)
    middle = sma(closes, p)
    upper: list[float | None] = [None] * n
    lower: list[float | None] = [None] * n
    if n < p:
        return middle, upper, lower
    for i in range(p - 1, n):
        m = middle[i]
        var = sum((closes[i - j] - m) ** 2 for j in range(p)) / p
        std = var ** 0.5
        upper[i] = m + k * std
        lower[i] = m - k * std
    return middle, upper, lower
