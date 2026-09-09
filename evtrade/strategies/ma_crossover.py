from __future__ import annotations
"""示例向量化策略: 双均线交叉 (CuPy 统一 CPU/GPU 路径)

================================================================
✅  可改层 (strategies 子包)  ✅  —— 向量化策略抄这个文件改
================================================================
纯数组算子策略, 用 xp (numpy 或 cupy) 写, 一份代码跑 CPU/GPU:
  - fast EMA 上穿 slow EMA -> BUY
  - fast EMA 下穿 slow EMA -> SELL
  - 每桶至多一个信号 (交叉点)

EMA 用 indicators.xp_ema (统一 source of truth)。
"""
from ..indicators import xp_ema
from .vectorized_base import VectorizedStrategy, register_strategy


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
        fast = xp_ema(xp, close, fast_p)
        slow = xp_ema(xp, close, slow_p)

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
