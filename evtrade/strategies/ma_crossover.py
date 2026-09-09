from __future__ import annotations
"""示例向量化策略: 双均线交叉 (CuPy 统一 CPU/GPU 路径)

================================================================
✅  可改层 (strategies 子包)  ✅  —— 向量化策略抄这个文件改
================================================================
纯数组算子策略, 用 xp (numpy 或 cupy) 写, 一份代码跑 CPU/GPU:
  - fast EMA 上穿 slow EMA -> BUY
  - fast EMA 下穿 slow EMA -> SELL
  - 每桶至多一个信号 (交叉点)

与 channel_deviation (DSL/numba, 逐 bar 状态机) 的区别:
  - 本策略无跨桶持久状态机, 信号仅依赖 fast/slow EMA 的当前与上一桶值
  - EMA 递推虽是顺序的, 但用 xp 写后 cupy 在 device 上跑 (正确性锁定,
    性能优化可后续用 cumsum 近似或 RawKernel)
"""
from .base import register_strategy
from .vectorized_base import VectorizedStrategy


def _ema_xp(xp, values, p: int):
    """EMA 批量版 (xp 兼容; 与 indicators.ema.ema 同式)

    SMA seed = 前 p 个均值; 之后 EMA_t = price*k + EMA_{t-1}*(1-k), k=2/(p+1)。
    前 p-1 个填 NaN (未就绪)。递推用 Python 循环 (EMA 是顺序依赖, 无法纯向量化;
    cupy 上循环仍在 device 数组上逐元素写, 正确性优先, 性能后续可优化)。
    """
    n = len(values)
    out = xp.full(n, xp.nan, dtype=xp.float64)
    if n < p:
        return out
    k = 2.0 / (p + 1.0)
    # SMA seed (前 p 个均值)
    seed = xp.sum(values[:p]) / p
    out[p - 1] = seed
    e = float(seed)
    # 递推 (顺序; cupy 数组逐元素赋值)
    for i in range(p, n):
        e = float(values[i]) * k + e * (1.0 - k)
        out[i] = e
    return out


@register_strategy("ma_crossover")
class MACrossoverStrategy(VectorizedStrategy):
    """双均线交叉策略 (向量化, CuPy 路径)

    params (dict):
      fast: 快线 EMA 周期 (桶数)
      slow: 慢线 EMA 周期 (桶数)
    """

    params_spec = {
        "fast": {"default": 5,  "type": int, "min": 2, "max": 1000},
        "slow": {"default": 20, "type": int, "min": 2, "max": 1000},
    }

    def compute_signals(self, xp, bars: dict, params: dict):
        fast_p = int(params["fast"])
        slow_p = int(params["slow"])
        close = bars["c"]
        fast = _ema_xp(xp, close, fast_p)
        slow = _ema_xp(xp, close, slow_p)

        # 交叉检测: fast-slow 符号变化
        diff = fast - slow
        # 上一桶的 diff (首桶无前值, 用 0 占位 -> 不产生信号)
        prev_diff = xp.empty_like(diff)
        prev_diff[1:] = diff[:-1]
        prev_diff[0] = 0.0

        # NaN 处理: EMA 未就绪时 diff=NaN, 比较得 False (不产信号)
        buy = (diff > 0) & (prev_diff <= 0) & xp.isfinite(diff) & (bars["mark"] == 1)
        sell = (diff < 0) & (prev_diff >= 0) & xp.isfinite(diff) & (bars["mark"] == 1)

        # 同根 bar buy/sell 互斥 (符号从 0 转 +/- 才算交叉; 不会同时成立)
        sig = xp.where(buy, 1, xp.where(sell, -1, 0)).astype(xp.int8)
        return sig


# 注册 sweep 网格允许的参数名
from ..core.sweep import register_grid_keys
register_grid_keys(set(MACrossoverStrategy.params_spec.keys()))
