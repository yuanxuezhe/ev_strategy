from __future__ import annotations
"""ATR (Average True Range) 指标 (纯函数批量 + @njit 增量版; DSL 三端可调)

================================================================
✅  可改层 (indicators 子包)  ✅
================================================================
True Range = max(H-L, |H-prev_close|, |L-prev_close|)
ATR(p) = TR 的 p 期 SMA (简单平均) 或 EMA (指数平均)。
本实现默认 SMA (与 MetaTrader/TradingView 一致)。

@njit 增量版 (DSL 三端可调):
  - atr_push(state, h, l, c, p) -> state  (Wilder 平滑: k=1/p)
  - atr_current(state, h, l, c, p) -> float
state = (sum: float, count: int, prev_close: float, atr: float)
"""
from typing import Sequence

from numba import njit


# ============ 纯函数版 (复盘 / jupyter 友好) ============

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


# ============ @njit 增量版 (DSL 三端可调; Wilder 平滑 k=1/p) ============

@njit(cache=True)
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
    # 第一根无前 close: TR = H - L
    if s_count == 0:
        tr = h - l
    else:
        diff1 = h - l
        diff2 = h - prev_close
        if diff2 < 0.0:
            diff2 = -diff2
        diff3 = l - prev_close
        if diff3 < 0.0:
            diff3 = -diff3
        tr = diff1
        if diff2 > tr:
            tr = diff2
        if diff3 > tr:
            tr = diff3
    if s_count < p:
        s_sum += tr
        s_count += 1
        if s_count == p:
            s_atr = s_sum / p
    else:
        s_atr = s_atr * ((p - 1.0) / p) + tr / p
        s_count += 1
    return s_sum, s_count, c, s_atr


@njit(cache=True)
def atr_current(state, h, l, c, p):
    """ATR 当前值 (假设当前未闭合桶为 (h, l, c), 不改 state); 数据不足返回 0.0"""
    s_sum = state[0]
    s_count = state[1]
    prev_close = state[2]
    s_atr = state[3]
    if s_count == 0:
        tr = h - l
    else:
        diff1 = h - l
        diff2 = h - prev_close
        if diff2 < 0.0:
            diff2 = -diff2
        diff3 = l - prev_close
        if diff3 < 0.0:
            diff3 = -diff3
        tr = diff1
        if diff2 > tr:
            tr = diff2
        if diff3 > tr:
            tr = diff3
    if s_count < p - 1:
        return 0.0
    if s_count == p - 1:
        return (s_sum + tr) / p
    return s_atr * ((p - 1.0) / p) + tr / p


# ============ CUDA __device__ 源码 (DSL CUDA 渲染器内联) ============
# ATR push 返回 4 标量 (sum, count, prev_close, atr); struct return

CUDA_DEVICE_ATR_PUSH = r"""
// __device__ atr_push (in: s_sum, s_count, prev_close, s_atr, h, l, c, p) -> ATRStateRet
struct ATRStateRet { double sum; long long count; double prev_close; double atr; };
__device__ __forceinline__ ATRStateRet atr_push(double s_sum, long long s_count, double prev_close, double s_atr,
                                                  double h, double l, double c, long long p) {
    ATRStateRet r;
    double tr;
    if (s_count == 0) {
        tr = h - l;
    } else {
        double d1 = h - l;
        double d2 = h - prev_close; if (d2 < 0.0) d2 = -d2;
        double d3 = l - prev_close; if (d3 < 0.0) d3 = -d3;
        tr = d1;
        if (d2 > tr) tr = d2;
        if (d3 > tr) tr = d3;
    }
    if (s_count < p) {
        r.sum = s_sum + tr;
        r.count = s_count + 1;
        r.prev_close = c;
        r.atr = (r.count == p) ? (r.sum / (double)p) : s_atr;
    } else {
        r.sum = s_sum;
        r.count = s_count + 1;
        r.prev_close = c;
        r.atr = s_atr * ((double)(p - 1) / (double)p) + tr / (double)p;
    }
    return r;
}
"""

CUDA_DEVICE_ATR_CURRENT = r"""
// __device__ atr_current (state, h, l, c, p) -> atr_now (0.0 if not ready)
__device__ __forceinline__ double atr_current(double s_sum, long long s_count, double prev_close, double s_atr,
                                                double h, double l, double c, long long p) {
    double tr;
    if (s_count == 0) {
        tr = h - l;
    } else {
        double d1 = h - l;
        double d2 = h - prev_close; if (d2 < 0.0) d2 = -d2;
        double d3 = l - prev_close; if (d3 < 0.0) d3 = -d3;
        tr = d1;
        if (d2 > tr) tr = d2;
        if (d3 > tr) tr = d3;
    }
    if (s_count < p - 1) return 0.0;
    if (s_count == p - 1) return (s_sum + tr) / (double)p;
    return s_atr * ((double)(p - 1) / (double)p) + tr / (double)p;
}
"""
