from __future__ import annotations
"""RSI (Relative Strength Index) 指标

RSI(p) = 100 - 100 / (1 + RS)
RS = 平均涨幅 / 平均跌幅 (Wilder 平滑: 第一个 RSI 用 SMA, 之后 EMA-like)
本实现采用 Wilder 标准平滑 (与 TradingView/MT4 一致)。

step API (策略 step 调):
  - rsi_step(state: RSIState, close, p) -> (RSIState, rsi)
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class RSIState:
    """RSI 增量 state (Wilder 平滑); rsi_step in/out"""
    sum_g: float = 0.0
    sum_l: float = 0.0
    avg_g: float = 0.0
    avg_l: float = 0.0
    count: int = 0
    prev_close: float = 0.0


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


def rsi_step(state: RSIState, close: float, p: int) -> tuple[RSIState, float]:
    """RSI 单步: state + close -> (new_state, rsi)

    规则 (Wilder 平滑):
      - count == 0: prev_close=close, count=1, ret=0 (数据不足)
      - count >= 1: g = max(close-prev_close, 0); l = max(prev_close-close, 0)
        count < p+1: sum += g/l; count++; 若 count==p+1: avg = sum/p, ret=100-100/(1+avg_g/avg_l)
        count >= p+1: avg = avg*(p-1)/p + new/p; count++; ret=100-100/(1+avg_g/avg_l)
    """
    if state.count == 0:
        return RSIState(prev_close=close, count=1), 0.0
    diff = close - state.prev_close
    g = diff if diff > 0.0 else 0.0
    l = -diff if diff < 0.0 else 0.0
    if state.count <= p:
        new_sum_g = state.sum_g + g
        new_sum_l = state.sum_l + l
        new_count = state.count + 1
        if new_count < p + 1:
            return RSIState(sum_g=new_sum_g, sum_l=new_sum_l,
                            count=new_count, prev_close=close), 0.0
        # new_count == p + 1: SMA seed
        avg_g = new_sum_g / p
        avg_l = new_sum_l / p
        rsi_v = 100.0 if avg_l == 0.0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
        return RSIState(sum_g=new_sum_g, sum_l=new_sum_l,
                        avg_g=avg_g, avg_l=avg_l,
                        count=new_count, prev_close=close), rsi_v
    # count >= p+1: Wilder 平滑
    avg_g = (state.avg_g * (p - 1) + g) / p
    avg_l = (state.avg_l * (p - 1) + l) / p
    rsi_v = 100.0 if avg_l == 0.0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return RSIState(sum_g=state.sum_g, sum_l=state.sum_l,
                    avg_g=avg_g, avg_l=avg_l,
                    count=state.count + 1, prev_close=close), rsi_v