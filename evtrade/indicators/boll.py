from __future__ import annotations
"""Bollinger Bands (布林带) 指标 (纯函数批量 + @njit 增量版; DSL 三端可调)

================================================================
✅  可改层 (indicators 子包)  ✅
================================================================
中轨 = MA(N) (简单移动平均)
上轨 = 中轨 + k * Std(N)
下轨 = 中轨 - k * Std(N)
默认 N=20, k=2 (与 TradingView 一致)。

@njit 增量版 (DSL 三端可调):
  - sma_push(state, value) -> state; sma_current(state, pending, p) -> float
  - boll_push(state, value) -> state; boll_current(state, pending, p, k) -> (mid, upper, lower)
简化: 增量版用 running mean (sum/count), 不严格 sliding window SMA (避免
@njit 维护 buffer)。批量校核请用 sma(values, p) 纯函数版。
state:
  sma:  (sum: float, count: int)
  boll: (sum: float, sumsq: float, count: int)
"""
from typing import Sequence

from numba import njit


# ============ 纯函数版 (复盘 / jupyter 友好) ============

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


# ============ @njit 增量版 (DSL 三端可调; running mean) ============

@njit(cache=True)
def sma_push(state, value):
    """SMA running-mean 增量推入: state=(sum, count)"""
    s_sum = state[0]
    count = state[1]
    return s_sum + value, count + 1


@njit(cache=True)
def sma_current(state, pending, p):
    """SMA 当前值 (含 pending, 不改 state); 数据不足返回 0.0

    表达式:
      count == 0: 0.0
      count < p-1: 0.0
      count == p-1: (sum + pending) / p  (凑齐 p 个)
      count >= p: (sum + pending) / (count + 1)  (running mean)
    """
    s_sum = state[0]
    count = state[1]
    if count < p - 1:
        return 0.0
    if count == p - 1:
        return (s_sum + pending) / p
    return (s_sum + pending) / (count + 1)


@njit(cache=True)
def boll_push(state, value):
    """Bollinger 增量推入: state=(sum, sumsq, count)"""
    s_sum = state[0]
    s_sumsq = state[1]
    count = state[2]
    return s_sum + value, s_sumsq + value * value, count + 1


@njit(cache=True)
def boll_current(state, pending, p, k):
    """Bollinger 当前值 (含 pending, 不改 state); 返回 (mid, upper, lower)

    数据不足时 (count < p-1) 返回 (0.0, 0.0, 0.0)。
    """
    s_sum = state[0]
    s_sumsq = state[1]
    count = state[2]
    if count < p - 1:
        return 0.0, 0.0, 0.0
    new_sum = s_sum + pending
    new_sumsq = s_sumsq + pending * pending
    new_count = count + 1
    mid = new_sum / new_count
    var = new_sumsq / new_count - mid * mid
    if var < 0.0:
        var = 0.0   # 数值误差保护
    std = var ** 0.5
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower


# ============ CUDA __device__ 源码 (DSL CUDA 渲染器内联) ============

CUDA_DEVICE_SMA_PUSH = r"""
// __device__ sma_push (in: s_sum, count, value) -> SMAStateRet
struct SMAStateRet { double sum; long long count; };
__device__ __forceinline__ SMAStateRet sma_push(double s_sum, long long count, double value) {
    SMAStateRet r;
    r.sum = s_sum + value;
    r.count = count + 1;
    return r;
}
"""

CUDA_DEVICE_SMA_CURRENT = r"""
// __device__ sma_current (state=(sum,count), pending, p) -> sma_now (0.0 if not ready)
__device__ __forceinline__ double sma_current(double s_sum, long long count, double pending, long long p) {
    if (count < p - 1) return 0.0;
    if (count == p - 1) return (s_sum + pending) / (double)p;
    return (s_sum + pending) / (double)(count + 1);
}
"""

CUDA_DEVICE_BOLL_PUSH = r"""
// __device__ boll_push (in: s_sum, s_sumsq, count, value) -> BollStateRet
struct BollStateRet { double sum; double sumsq; long long count; };
__device__ __forceinline__ BollStateRet boll_push(double s_sum, double s_sumsq, long long count, double value) {
    BollStateRet r;
    r.sum = s_sum + value;
    r.sumsq = s_sumsq + value * value;
    r.count = count + 1;
    return r;
}
"""

CUDA_DEVICE_BOLL_CURRENT = r"""
// __device__ boll_current (state=(sum,sumsq,count), pending, p, k) -> (mid, upper, lower)
// 返回 3 个 double; 用结构体 out 传出。
struct BollOut { double mid, upper, lower; };
__device__ __forceinline__ BollOut boll_current(double s_sum, double s_sumsq, long long count,
                                                  double pending, long long p, double k) {
    BollOut r;
    if (count < p - 1) { r.mid = 0.0; r.upper = 0.0; r.lower = 0.0; return r; }
    double new_sum = s_sum + pending;
    double new_sumsq = s_sumsq + pending * pending;
    double new_count_d = (double)(count + 1);
    double mid = new_sum / new_count_d;
    double var = new_sumsq / new_count_d - mid * mid;
    if (var < 0.0) var = 0.0;
    double std = sqrt(var);
    r.mid = mid;
    r.upper = mid + k * std;
    r.lower = mid - k * std;
    return r;
}
"""
