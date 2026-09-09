from __future__ import annotations
"""RSI (Relative Strength Index) 指标 (纯 ndarray 版 + 纯 Python 增量版)

================================================================
✅  可改层 (indicators 子包)  ✅
================================================================
RSI(p) = 100 - 100 / (1 + RS)
RS = 平均涨幅 / 平均跌幅 (Wilder 平滑: 第一个 RSI 用 SMA, 之后 EMA-like)
本实现采用 Wilder 标准平滑 (与 TradingView/MT4 一致)。

@Python 增量版 (Engine 路径调用):
  - rsi_push(state, close, p) -> state
  - rsi_current(state, close, p) -> float
state = (sum_g, sum_l, avg_g, avg_l, count, prev_close) 6-tuple;
  - count < 2: 无 gain/loss
  - count <= p: 累加 sum_g/sum_l
  - count == p+1: SMA seed avg = sum/p
  - count > p+1: Wilder 平滑 avg = avg*(p-1)/p + new/p
"""
import numpy as np


# ============ 纯函数版 (复盘 / jupyter 友好, 返 ndarray) ============

def rsi(closes, p: int = 14):
    """Wilder RSI(p); 返回长度 == len(closes); 前 p 个为 NaN"""
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= p:
        return out
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff
    # 第一个 RSI: 用前 p 个 gain/loss 的 SMA
    avg_g = float(np.sum(gains[1:p + 1])) / p
    avg_l = float(np.sum(losses[1:p + 1])) / p
    out[p] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    # Wilder 平滑
    for i in range(p + 1, n):
        avg_g = (avg_g * (p - 1) + gains[i]) / p
        avg_l = (avg_l * (p - 1) + losses[i]) / p
        out[i] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return out


# ============ 纯 Python 增量版 (Engine.on_bars 路径; Wilder 平滑) ============

def rsi_push(state, close, p):
    """RSI 增量推入 (Wilder 平滑): state=(sum_g, sum_l, avg_g, avg_l, count, prev_close)

    表达式:
      count == 0: prev_close=close, count=1
      count >= 1: g = max(close-prev_close, 0); l = max(prev_close-close, 0)
        count <= p: sum += g/l; count++; if count==p+1: avg = sum/p
        count > p+1: avg = avg*(p-1)/p + new/p; count++
    返回新 state。
    """
    sum_g = state[0]
    sum_l = state[1]
    avg_g = state[2]
    avg_l = state[3]
    count = state[4]
    prev_close = state[5]
    if count == 0:
        return sum_g, sum_l, avg_g, avg_l, 1, close
    diff = close - prev_close
    g = diff if diff > 0.0 else 0.0
    l = -diff if diff < 0.0 else 0.0
    if count <= p:
        sum_g += g
        sum_l += l
        count += 1
        if count == p + 1:
            avg_g = sum_g / p
            avg_l = sum_l / p
    else:
        avg_g = (avg_g * (p - 1) + g) / p
        avg_l = (avg_l * (p - 1) + l) / p
        count += 1
    return sum_g, sum_l, avg_g, avg_l, count, close


def rsi_current(state, close, p):
    """RSI 当前值 (假设当前未闭合 close, 不改 state); 数据不足返回 0.0"""
    sum_g = state[0]
    sum_l = state[1]
    avg_g = state[2]
    avg_l = state[3]
    count = state[4]
    prev_close = state[5]
    if count < 2 or count < p + 1:
        return 0.0
    diff = close - prev_close
    g = diff if diff > 0.0 else 0.0
    l = -diff if diff < 0.0 else 0.0
    avg_g_now = (avg_g * (p - 1) + g) / p
    avg_l_now = (avg_l * (p - 1) + l) / p
    if avg_l_now == 0.0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_g_now / avg_l_now)
