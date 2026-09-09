from __future__ import annotations
"""RSI (Relative Strength Index) 指标 (纯函数批量 + @njit 增量版; DSL 三端可调)

================================================================
✅  可改层 (indicators 子包)  ✅
================================================================
RSI(p) = 100 - 100 / (1 + RS)
RS = 平均涨幅 / 平均跌幅 (Wilder 平滑: 第一个 RSI 用 SMA, 之后 EMA-like)
本实现采用 Wilder 标准平滑 (与 TradingView/MT4 一致)。

@njit 增量版 (DSL 三端可调):
  - rsi_push(state, close, p) -> state
  - rsi_current(state, close, p) -> float
state = (sum_g, sum_l, avg_g, avg_l, count, prev_close) 6-tuple;
  - count < 2: 无 gain/loss
  - count <= p: 累加 sum_g/sum_l
  - count == p+1: SMA seed avg = sum/p
  - count > p+1: Wilder 平滑 avg = avg*(p-1)/p + new/p
"""
from typing import Sequence

from numba import njit


# ============ 纯函数版 (复盘 / jupyter 友好) ============

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


# ============ @njit 增量版 (DSL 三端可调; Wilder 平滑) ============

@njit(cache=True)
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


@njit(cache=True)
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
    # Wilder 递推一次
    if count == p + 1:
        # avg_g 此刻 = sum_g/p (SMA seed); pending 是第 p+1 对, 用 Wilder 公式递推
        avg_g_now = (avg_g * (p - 1) + g) / p
        avg_l_now = (avg_l * (p - 1) + l) / p
    else:
        avg_g_now = (avg_g * (p - 1) + g) / p
        avg_l_now = (avg_l * (p - 1) + l) / p
    if avg_l_now == 0.0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_g_now / avg_l_now)


# ============ CUDA __device__ 源码 (DSL CUDA 渲染器内联) ============

CUDA_DEVICE_RSI_PUSH = r"""
// __device__ rsi_push (in: state, close, p) -> RSIStateRet
struct RSIStateRet { double sum_g; double sum_l; double avg_g; double avg_l; long long count; double prev_close; };
__device__ __forceinline__ RSIStateRet rsi_push(double sum_g, double sum_l, double avg_g, double avg_l,
                                                  long long count, double prev_close, double close, long long p) {
    RSIStateRet r;
    if (count == 0) {
        r.sum_g = sum_g; r.sum_l = sum_l; r.avg_g = avg_g; r.avg_l = avg_l;
        r.count = 1; r.prev_close = close;
        return r;
    }
    double diff = close - prev_close;
    double g = (diff > 0.0) ? diff : 0.0;
    double l = (diff < 0.0) ? -diff : 0.0;
    if (count <= p) {
        r.sum_g = sum_g + g;
        r.sum_l = sum_l + l;
        r.count = count + 1;
        r.prev_close = close;
        if (r.count == p + 1) {
            r.avg_g = r.sum_g / (double)p;
            r.avg_l = r.sum_l / (double)p;
        } else {
            r.avg_g = avg_g; r.avg_l = avg_l;
        }
    } else {
        r.sum_g = sum_g; r.sum_l = sum_l;
        r.avg_g = (avg_g * (double)(p - 1) + g) / (double)p;
        r.avg_l = (avg_l * (double)(p - 1) + l) / (double)p;
        r.count = count + 1;
        r.prev_close = close;
    }
    return r;
}
"""

CUDA_DEVICE_RSI_CURRENT = r"""
// __device__ rsi_current (state, close, p) -> rsi_now (0.0 if not ready)
__device__ __forceinline__ double rsi_current(double sum_g, double sum_l, double avg_g, double avg_l,
                                                long long count, double prev_close, double close, long long p) {
    if (count < 2 || count < p + 1) return 0.0;
    double diff = close - prev_close;
    double g = (diff > 0.0) ? diff : 0.0;
    double l = (diff < 0.0) ? -diff : 0.0;
    double avg_g_now = (avg_g * (double)(p - 1) + g) / (double)p;
    double avg_l_now = (avg_l * (double)(p - 1) + l) / (double)p;
    if (avg_l_now == 0.0) return 100.0;
    return 100.0 - 100.0 / (1.0 + avg_g_now / avg_l_now);
}
"""
