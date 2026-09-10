"""MACrossoverStrategy: 双均线交叉策略 (stateful step)

EMA fast 上穿/slow -> BUY; 下穿 -> SELL; 每桶至多一个信号。

参数:
  fast  fast EMA 周期
  slow  slow EMA 周期
"""
from dataclasses import dataclass, field

from ..indicators import EMAState, ema_step
from .vectorized_base import VectorizedStrategy, register_strategy


@dataclass
class MACrossoverState:
    """策略持久状态: fast EMA + slow EMA + 上一桶 diff"""
    fast: EMAState = field(default_factory=EMAState)
    slow: EMAState = field(default_factory=EMAState)
    prev_diff: float = 0.0
    has_prev: bool = False


@register_strategy("ma_crossover")
class MACrossoverStrategy(VectorizedStrategy):
    """双均线交叉策略"""

    params_spec = {
        "fast": {"default": 5,  "type": int, "min": 2, "max": 1000},
        "slow": {"default": 20, "type": int, "min": 2, "max": 1000},
    }

    def init_state(self, params: dict) -> MACrossoverState:
        return MACrossoverState()

    def step(self, state: MACrossoverState, bar: dict, params: dict
             ) -> tuple[MACrossoverState, int]:
        fast_p = int(params["fast"])
        slow_p = int(params["slow"])
        close = float(bar["c"])

        # 推 EMA (mark=0 也推, 让 state 累积; 与原版 vectorized 路径语义一致)
        state.fast, fast = ema_step(state.fast, close, fast_p)
        state.slow, slow = ema_step(state.slow, close, slow_p)

        # 预热段 / fast 还没就绪: 不产信号; fast 未就绪时清 prev_diff 避免污染下一桶
        if bar["mark"] == 0:
            return state, 0
        if state.fast.count < fast_p:
            state.prev_diff = 0.0
            state.has_prev = True
            return state, 0

        # 交叉检测: diff 符号变化
        diff = fast - slow
        sig = 0
        if state.has_prev:
            if diff > 0 and state.prev_diff <= 0:
                sig = 1   # BUY: 上穿
            elif diff < 0 and state.prev_diff >= 0:
                sig = -1  # SELL: 下穿

        state.prev_diff = diff
        state.has_prev = True
        return state, sig
