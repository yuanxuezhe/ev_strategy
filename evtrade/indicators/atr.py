from __future__ import annotations
"""ATR (Average True Range) 指标

True Range = max(H-L, |H-prev_close|, |L-prev_close|)
ATR(p) = TR 的 p 期 SMA (默认) 或 EMA。
本实现默认 SMA (与 MetaTrader/TradingView 一致)。

step API (策略 step 调):
  - atr_step(state: ATRState, h, l, c, p) -> (ATRState, atr)
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ATRState:
    """ATR 增量 state; atr_step in/out (Wilder 平滑: k=1/p)"""
    sum: float = 0.0
    count: int = 0
    prev_close: float = 0.0
    atr: float = 0.0


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


def _tr(h: float, l: float, prev_close: float, has_prev: bool) -> float:
    """单根 true range; has_prev=False 时 (首根) TR = h - l"""
    if not has_prev:
        return h - l
    d1 = h - l
    d2 = abs(h - prev_close)
    d3 = abs(l - prev_close)
    tr = d1
    if d2 > tr:
        tr = d2
    if d3 > tr:
        tr = d3
    return tr


def atr_step(state: ATRState, h: float, l: float, c: float,
             p: int) -> tuple[ATRState, float]:
    """ATR 单步: state + (h, l, c) -> (new_state, atr)

    规则 (Wilder 平滑, k=1/p):
      - TR = _tr(h, l, state.prev_close, state.count > 0)
      - count < p:   sum += TR; count += 1; if count==p: atr = sum/p
      - count >= p:  atr = atr * (p-1)/p + TR/p; count += 1
      - new_state.prev_close = c
    """
    has_prev = state.count > 0
    tr = _tr(h, l, state.prev_close, has_prev)
    if state.count < p:
        new_sum = state.sum + tr
        new_count = state.count + 1
        if new_count < p:
            return ATRState(sum=new_sum, count=new_count,
                            prev_close=c, atr=0.0), 0.0
        new_atr = new_sum / p
        return ATRState(sum=new_sum, count=new_count,
                        prev_close=c, atr=new_atr), new_atr
    # count >= p: Wilder 平滑
    new_atr = state.atr * ((p - 1.0) / p) + tr / p
    return ATRState(sum=state.sum, count=state.count + 1,
                    prev_close=c, atr=new_atr), new_atr