from __future__ import annotations
"""EMA / EMAChannel 纯函数 (批量计算, 复盘 / jupyter 友好)

================================================================
✅  可改层 (indicators 子包)  ✅
================================================================
evtrade.kernel.IncrementalEMA 是热路径用 (逐根 O(1));
本模块是**纯函数版** (一次性批量), 用于复盘 / 报告 / 单元测试。

两种实现口径一致: 同一份浮点表达式, 同一份 SMA seed。
================================================================
"""
from typing import Sequence


def ema(values: Sequence[float], p: int) -> list[float | None]:
    """EMA(values, p): 输入长度 >= p 时返回完整序列 (前 p-1 个为 None, 之后为递推值)

    SMA seed = 前 p 个均值; 之后 EMA_t = price * k + EMA_{t-1} * (1-k), k = 2/(p+1)

    返回长度 == len(values); len(values) < p 时全部为 None。
    """
    n = len(values)
    out: list[float | None] = [None] * n
    if n < p:
        return out
    k = 2.0 / (p + 1.0)
    seed = sum(values[:p]) / p
    out[p - 1] = seed
    e = seed
    for i in range(p, n):
        v = values[i]
        e = v * k + e * (1.0 - k)
        out[i] = e
    return out


def ema_channel(highs: Sequence[float], lows: Sequence[float], p: int) -> tuple[list[float | None], list[float | None]]:
    """通达信蓝通道轨: 上轨 = EMA(H, p), 下轨 = EMA(L, p)"""
    return ema(highs, p), ema(lows, p)
