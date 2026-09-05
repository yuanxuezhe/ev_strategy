from __future__ import annotations
"""RSI (Relative Strength Index) 纯函数

================================================================
✅  可改层 (indicators 子包)  ✅
================================================================
RSI(p) = 100 - 100 / (1 + RS)
RS = 平均涨幅 / 平均跌幅 (Wilder 平滑: 第一个 RSI 用 SMA, 之后 EMA-like)
本实现采用 Wilder 标准平滑 (与 TradingView/MT4 一致)。
"""
from typing import Sequence


def rsi(closes: Sequence[float], p: int = 14) -> list[float | None]:
    """Wilder RSI(p); 返回长度 == len(closes); 前 p 个为 None"""
    n = len(closes)
    out: list[float | None] = [None] * n
    if n <= p:
        return out
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff
    # 第一个 RSI: 用前 p 个 gain/loss 的 SMA
    avg_g = sum(gains[1:p + 1]) / p
    avg_l = sum(losses[1:p + 1]) / p
    out[p] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    # Wilder 平滑
    for i in range(p + 1, n):
        avg_g = (avg_g * (p - 1) + gains[i]) / p
        avg_l = (avg_l * (p - 1) + losses[i]) / p
        out[i] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return out
