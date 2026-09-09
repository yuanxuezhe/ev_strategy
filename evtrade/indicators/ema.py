from __future__ import annotations
"""EMA / EMAChannel 指标 (纯 xp 版 + 纯 Python 增量版)

================================================================
✅  可改层 (indicators 子包)  ✅
================================================================
两种形态:
  1. 纯 xp 版 (xp_ema, xp_ema_channel): 一次性批量算整段序列,
     接受 xp (numpy|cupy) 模块; 用于 compute_signals(xp, ...) 主体。
  2. 纯 Python 增量版 (ema_push, ema_current, ema_channel_push,
     ema_channel_current): 逐根 O(1) 维护增量 state, 供
     compute_signals_for_one_bar (Engine.on_bars) 逐 bar 路径调用;
     返回值与浮点表达式与批量版逐位一致。

@njit / CUDA __device__ 渲染器已下线 (DSL 整体废弃); 本文件不依赖 numba。

@Python 增量 API (Engine 路径调用):
  - ema_push(s_sum, s_count, s_ema, value, p) -> (s_sum', s_count', s_ema')
  - ema_current(s_sum, s_count, s_ema, pending, p) -> float
  - ema_channel_push(us, uc, ue, ds, dc, de, h, l, p) -> (us', uc', ue', ds', dc', de')
  - ema_channel_current(us, uc, ue, ds, dc, de, h, l, p) -> (up, dw)

state 三标量拆开传入 (无 tuple 字段; 与旧 numba jitclass 字段约定一致,
  refactor 后唯一来源即此处):
  - count < p: 累加 sum; ema 仍为 0.0
  - count == p: ema = sum / p  (SMA seed)
  - count > p: ema = value*k + ema*(1-k) (Wilder EMA 递推)
ema_current 返回:
  - count < p-1: 0.0 (未就绪)
  - count == p-1: (sum + pending) / p  (凑齐 p 个 -> SMA seed)
  - count >= p: pending*k + ema*(1-k) (递推一次)
"""
import numpy as np


# ============ 纯 xp 版 (compute_signals 主体调用) ============

def xp_ema(xp, values, p: int):
    """EMA 批量版 (xp 兼容); 前 p-1 根 NaN, 之后递推。

    SMA seed = 前 p 个均值; EMA_t = value*k + EMA_{t-1}*(1-k), k=2/(p+1)。
    """
    n = len(values)
    out = xp.full(n, xp.nan, dtype=xp.float64)
    if n < p:
        return out
    k = 2.0 / (p + 1.0)
    seed = xp.sum(values[:p]) / p
    out[p - 1] = seed
    e = float(seed)
    for i in range(p, n):
        e = float(values[i]) * k + e * (1.0 - k)
        out[i] = e
    return out


def xp_ema_channel(xp, highs, lows, p: int):
    """EMA 通道: 上轨 = EMA(H, p), 下轨 = EMA(L, p)"""
    return xp_ema(xp, highs, p), xp_ema(xp, lows, p)


# ============ 纯 Python 增量版 (Engine.on_bars 路径) ============

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


def ema_channel_push(us, uc, ue, ds, dc, de, h, l, p):
    """EMA 通道增量推入: up_st 推 h, dw_st 推 l; 返回 6 标量新 state"""
    us, uc, ue = ema_push(us, uc, ue, h, p)
    ds, dc, de = ema_push(ds, dc, de, l, p)
    return us, uc, ue, ds, dc, de


def ema_channel_current(us, uc, ue, ds, dc, de, h, l, p):
    """EMA 通道当前值: 算得 (up, dw) 当前轨, O(1)"""
    return ema_current(us, uc, ue, h, p), ema_current(ds, dc, de, l, p)


# ============ 纯函数版 (jupyter / 复盘, 返 ndarray) ============

def ema(values, p: int):
    """EMA(values, p): 输入长度 >= p 时返回完整 ndarray (前 p-1 个为 NaN, 之后为递推值)

    SMA seed = 前 p 个均值; 之后 EMA_t = value * k + EMA_{t-1} * (1-k), k = 2/(p+1)

    返回长度 == len(values); len(values) < p 时全部为 NaN。
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < p:
        return out
    k = 2.0 / (p + 1.0)
    seed = np.sum(values[:p]) / p
    out[p - 1] = seed
    e = float(seed)
    for i in range(p, n):
        e = float(values[i]) * k + e * (1.0 - k)
        out[i] = e
    return out


def ema_channel(highs, lows, p: int):
    """EMA 通道 (numpy): 上轨 = EMA(H, p), 下轨 = EMA(L, p)"""
    return ema(highs, p), ema(lows, p)
