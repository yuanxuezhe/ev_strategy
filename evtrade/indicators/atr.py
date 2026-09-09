from __future__ import annotations
"""ATR (Average True Range) 指标 (纯 xp / ndarray 版 + 纯 Python 增量版)

================================================================
✅  可改层 (indicators 子包)  ✅
================================================================
True Range = max(H-L, |H-prev_close|, |L-prev_close|)
ATR(p) = TR 的 p 期 SMA (简单平均) 或 EMA (指数平均)。
本实现默认 SMA (与 MetaTrader/TradingView 一致)。

@Python 增量版 (Engine 路径调用):
  - atr_push(state, h, l, c, p) -> state  (Wilder 平滑: k=1/p)
  - atr_current(state, h, l, c, p) -> float
state = (sum: float, count: int, prev_close: float, atr: float)
"""
import numpy as np


# ============ 纯函数版 (jupyter / 复盘, 返 ndarray) ============

def true_range(highs, lows, closes):
    """逐根 true range; 第一根 TR = H - L (无前 close), 之后 = max(H-L, |H-prev_C|, |L-prev_C|)

    返回长度 == len(highs) 的 ndarray; 输入同长度。
    """
    n = len(highs)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    out[0] = highs[0] - lows[0]
    for i in range(1, n):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        d1 = h - l
        d2 = h - pc if h - pc >= 0 else pc - h
        d3 = l - pc if l - pc >= 0 else pc - l
        out[i] = d1
        if d2 > out[i]:
            out[i] = d2
        if d3 > out[i]:
            out[i] = d3
    return out


def atr(highs, lows, closes, p: int = 14, ema: bool = False):
    """ATR(p): True Range 的 p 期平均

    ema=False (默认): SMA (简单移动平均) —— 与 MetaTrader/TradingView 一致
    ema=True:        EMA (指数平均), 用 indicators.ema
    """
    tr = true_range(highs, lows, closes)
    if not ema:
        # SMA: 前 p 根 TR 均值 -> 首个 ATR
        n = len(tr)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < p:
            return out
        out[p - 1] = np.nansum(tr[:p]) / p
        # 滑动窗口 SMA: out[i] = out[i-1] + (tr[i] - tr[i-p]) / p
        for i in range(p, n):
            ti = tr[i] if not np.isnan(tr[i]) else 0.0
            tip = tr[i - p] if not np.isnan(tr[i - p]) else 0.0
            out[i] = out[i - 1] + (ti - tip) / p
        return out
    else:
        from .ema import ema as _ema
        # tr 是 ndarray, _ema 内部用 float() 与 np.sum; 兼容 (NaN 视为 0)
        tr_filled = np.where(np.isnan(tr), 0.0, tr)
        return _ema(tr_filled, p)


# ============ 纯 Python 增量版 (Engine.on_bars 路径) ============

def atr_push(state, h, l, c, p):
    """ATR 增量推入 (Wilder 平滑): state=(sum, count, prev_close, atr)

    表达式:
      TR = max(h-l, |h-prev_close|, |l-prev_close|)
      count < p:  sum += TR; count++; if count==p: atr = sum/p
      count >= p: atr = atr*(p-1)/p + TR/p; count++
    返回新 state (含 prev_close <- c)。
    """
    s_sum = state[0]
    s_count = state[1]
    prev_close = state[2]
    s_atr = state[3]
    if s_count == 0:
        tr = h - l
    else:
        d1 = h - l
        d2 = h - prev_close
        if d2 < 0.0:
            d2 = -d2
        d3 = l - prev_close
        if d3 < 0.0:
            d3 = -d3
        tr = d1
        if d2 > tr:
            tr = d2
        if d3 > tr:
            tr = d3
    if s_count < p:
        s_sum += tr
        s_count += 1
        if s_count == p:
            s_atr = s_sum / p
    else:
        s_atr = s_atr * ((p - 1.0) / p) + tr / p
        s_count += 1
    return s_sum, s_count, c, s_atr


def atr_current(state, h, l, c, p):
    """ATR 当前值 (假设当前未闭合桶为 (h, l, c), 不改 state); 数据不足返回 0.0"""
    s_sum = state[0]
    s_count = state[1]
    prev_close = state[2]
    s_atr = state[3]
    if s_count == 0:
        tr = h - l
    else:
        d1 = h - l
        d2 = h - prev_close
        if d2 < 0.0:
            d2 = -d2
        d3 = l - prev_close
        if d3 < 0.0:
            d3 = -d3
        tr = d1
        if d2 > tr:
            tr = d2
        if d3 > tr:
            tr = d3
    if s_count < p - 1:
        return 0.0
    if s_count == p - 1:
        return (s_sum + tr) / p
    return s_atr * ((p - 1.0) / p) + tr / p
