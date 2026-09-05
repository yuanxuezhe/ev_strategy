from __future__ import annotations
"""ATR (Average True Range) 纯函数

================================================================
✅  可改层 (indicators 子包)  ✅
================================================================
True Range = max(H-L, |H-prev_close|, |L-prev_close|)
ATR(p) = TR 的 p 期 SMA (简单平均) 或 EMA (指数平均)。
本实现默认 SMA (与 MetaTrader/TradingView 一致)。
"""
from typing import Sequence


def true_range(highs: Sequence[float], lows: Sequence[float],
               closes: Sequence[float]) -> list[float | None]:
    """逐根 true range; 第一根 TR = H - L (无前 close), 之后 = max(H-L, |H-prev_C|, |L-prev_C|)"""
    n = len(highs)
    out: list[float | None] = [None] * n
    if n == 0:
        return out
    out[0] = highs[0] - lows[0]
    for i in range(1, n):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        out[i] = max(h - l, abs(h - pc), abs(l - pc))
    return out


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
        p: int = 14, ema: bool = False) -> list[float | None]:
    """ATR(p): True Range 的 p 期平均

    ema=False (默认): SMA (简单移动平均) —— 与 MetaTrader/TradingView 一致
    ema=True:        EMA (指数平均), 用 indicators.ema
    """
    tr = true_range(highs, lows, closes)
    if not ema:
        # SMA: 前 p 根 TR 均值 -> 首个 ATR
        n = len(tr)
        out: list[float | None] = [None] * n
        if n < p:
            return out
        out[p - 1] = sum(tr[:p]) / p  # type: ignore
        # 之后滑动窗口 SMA: out[i] = out[i-1] + (tr[i] - tr[i-p]) / p
        for i in range(p, n):
            out[i] = out[i - 1] + (tr[i] - tr[i - p]) / p  # type: ignore
        return out
    else:
        from .ema import ema as _ema
        return _ema(tr, p)  # type: ignore
