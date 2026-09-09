from __future__ import annotations
"""EMA / EMAChannel 指标 (纯函数批量 + @njit 增量版; DSL 三端可调)

================================================================
✅  可改层 (indicators 子包)  ✅
================================================================
两种形态:
  1. 纯函数版 (ema, ema_channel): 一次性批量算整段序列, 用于复盘 / jupyter / 单元测试
  2. @njit 增量版 (ema_push, ema_current, ema_channel_push, ema_channel_current):
     逐根 O(1) 维护增量状态, 供 DSL body (三端: Python exec / numba @njit / CUDA __device__)
     在策略内部调; 浮点表达式与原 _ema_push / _ema_current 逐位一致。

@njit 增量 API 形式 (DSL 可调):
  - ema_push(s_sum, s_count, s_ema, value, p) -> (s_sum', s_count', s_ema')
  - ema_current(s_sum, s_count, s_ema, pending, p) -> float
  - ema_channel_push(us, uc, ue, ds, dc, de, h, l, p) -> (us', uc', ue', ds', dc', de')
  - ema_channel_current(us, uc, ue, ds, dc, de, h, l, p) -> (up, dw)

state 三标量拆开传入 (无 tuple 字段; 兼容 numba jitclass 标量字段约束):
  - count < p: 累加 sum; ema 仍为 0.0
  - count == p: ema = sum / p  (SMA seed)
  - count > p: ema = value*k + ema*(1-k) (Wilder EMA 递推)
ema_current 返回:
  - count < p-1: 0.0 (未就绪)
  - count == p-1: (sum + pending) / p  (凑齐 p 个 -> SMA seed)
  - count >= p: pending*k + ema*(1-k) (递推一次)

新增/调整指标的扩展办法详见 kbs/11 §5 (DSL 增量 API + 白名单)。
"""
from typing import Sequence

from numba import njit


# ============ 纯函数版 (复盘 / jupyter 友好) ============

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


# ============ @njit 增量版 (DSL 三端可调) ============
# 浮点表达式与原 core/kernel._ema_push / _ema_current 逐位一致, 差分测试锁定
# (tests/test_differential.py::test_differential + tests/test_kernel_unit.py::test_ema_push_bitwise)。
# 配套 CUDA __device__ 版本见本文件末尾 (DSL CUDA 渲染器直接内联到 gpu kernel 模板)。
#
# 设计要点: state 三标量 (sum/count/ema) 拆开传入传出, 不传 tuple; 这样:
#   - numba jitclass 字段是 3 个独立标量 (兼容 jitclass 类型约束)
#   - DSL body 用 Python 多元赋值 `a, b, c = ema_push(a, b, c, v, p)` 自然调
#   - CUDA 端多元赋值展开为多语句 (device function 改用 in + ref out)

@njit(cache=True)
def ema_push(s_sum, s_count, s_ema, value, p):
    """EMA 增量推入: state=(s_sum, s_count, s_ema) 三标量, 返回新 (sum', count', ema')

    表达式: count < p 累加 sum; count == p 时 ema = sum/p (SMA seed);
    count > p 时 ema = value*k + ema*(1-k), k = 2/(p+1)。
    """
    if s_count < p:
        s_sum = s_sum + value
        s_count = s_count + 1
        if s_count == p:
            s_ema = s_sum / p
    else:
        k = 2.0 / (p + 1.0)
        s_ema = value * k + s_ema * (1.0 - k)
        s_count = s_count + 1
    return s_sum, s_count, s_ema


@njit(cache=True)
def ema_current(s_sum, s_count, s_ema, pending, p):
    """EMA 当前值 (不改 state); 数据不足返回 0.0

    表达式: count < p-1 返回 0.0; count == p-1 返回 (sum + pending)/p;
    count >= p 返回 pending*k + ema*(1-k), k = 2/(p+1)。
    """
    if s_count < p - 1:
        return 0.0
    if s_count == p - 1:
        return (s_sum + pending) / p
    k = 2.0 / (p + 1.0)
    return pending * k + s_ema * (1.0 - k)


@njit(cache=True)
def ema_channel_push(us, uc, ue, ds, dc, de, h, l, p):
    """EMA 通道增量推入: up_st 推 h, dw_st 推 l; 返回 6 标量新 state"""
    us, uc, ue = ema_push(us, uc, ue, h, p)
    ds, dc, de = ema_push(ds, dc, de, l, p)
    return us, uc, ue, ds, dc, de


@njit(cache=True)
def ema_channel_current(us, uc, ue, ds, dc, de, h, l, p):
    """EMA 通道当前值: 算得 (up, dw) 当前轨, O(1)"""
    return ema_current(us, uc, ue, h, p), ema_current(ds, dc, de, l, p)


# ============ CUDA __device__ 源码 (DSL CUDA 渲染器内联到 gpu kernel 模板) ============
# CUDA 端: device function 返回 struct (因 C++ 没有 Python tuple unpack 语法);
# DSL 多元赋值 (`a, b, c = func(...)`) 在 CUDA 端展开为
# 临时变量 (struct) + 多次赋值 (auto _t = func(...); a = _t.f0; b = _t.f1; c = _t.f2;)
# 浮点表达式与上面 @njit 版逐字一致 (k = 2.0 / (p + 1.0); sum/p; v*k + e*(1-k))。

CUDA_DEVICE_EMA_PUSH = r"""
// __device__ ema_push (in: s_sum, s_count, s_ema, value, p) -> EMAStateRet
// 表达式与 evtrade.indicators.ema.ema_push 逐字一致
struct EMAStateRet { double sum; long long count; double ema; };
__device__ __forceinline__ EMAStateRet ema_push(double s_sum, long long s_count, double s_ema,
                                                  double value, long long p) {
    EMAStateRet r;
    if (s_count < p) {
        r.sum = s_sum + value;
        r.count = s_count + 1;
        r.ema = (r.count == p) ? (r.sum / (double)p) : s_ema;
    } else {
        double k = 2.0 / ((double)p + 1.0);
        r.sum = s_sum;
        r.count = s_count + 1;
        r.ema = value * k + s_ema * (1.0 - k);
    }
    return r;
}
"""

CUDA_DEVICE_EMA_CHANNEL_PUSH = r"""
// __device__ ema_channel_push (in: us, uc, ue, ds, dc, de, h, l, p) -> EMAChannelRet
struct EMAChannelRet { EMAStateRet up; EMAStateRet dw; };
__device__ __forceinline__ EMAChannelRet ema_channel_push(double us, long long uc, double ue,
                                                            double ds, long long dc, double de,
                                                            double h, double l, long long p) {
    EMAChannelRet r;
    r.up = ema_push(us, uc, ue, h, p);
    r.dw = ema_push(ds, dc, de, l, p);
    return r;
}
"""

CUDA_DEVICE_EMA_CURRENT = r"""
// __device__ ema_current (in: state + pending, p) -> ema_now (0.0 if not ready)
// 表达式与 evtrade.indicators.ema.ema_current 逐字一致
__device__ __forceinline__ double ema_current(double s_sum, long long s_count, double s_ema,
                                                double pending, long long p) {
    if (s_count < p - 1) return 0.0;
    if (s_count == p - 1) return (s_sum + pending) / (double)p;
    double k = 2.0 / ((double)p + 1.0);
    return pending * k + s_ema * (1.0 - k);
}
"""
