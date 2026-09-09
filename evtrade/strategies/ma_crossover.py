from __future__ import annotations
"""示例策略: 双均线交叉 (strategy-step-only)

================================================================
✅  可改层 (strategies 子包)  ✅  —— 向量化策略抄这个文件改
================================================================
stateful 策略: 持 EMA fast/slow + 上一桶 diff;
step(state, bar, params) -> (state, sig):
  - mark=0 预热段返 0
  - fast EMA 上穿 slow EMA -> BUY
  - fast EMA 下穿 slow EMA -> SELL
  - 每桶至多一个信号 (交叉点)

EMA 用 indicators.ema_step (统一 source of truth, dataclass state)。
"""
from dataclasses import dataclass, field

from ..indicators import EMAState, ema_step
from .vectorized_base import VectorizedStrategy, register_strategy


# ============ 状态 dataclass ============

@dataclass
class MACrossoverState:
    """策略持久状态: fast EMA + slow EMA + 上一桶 diff"""
    fast:EMAState = field(default_factory=EMAState)
    slow:EMAState = field(default_factory=EMAState)
    prev_diff: float = 0.0
    has_prev: bool = False


# ============ 策略类 ============

@register_strategy("ma_crossover")
class MACrossoverStrategy(VectorizedStrategy):
    """双均线交叉策略 (stateful step)"""

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

        # 推 EMA (mark=0 也推, 让 state 累积; 与原版 xp_ema 整段算语义一致)
        state.fast, fast = ema_step(state.fast, close, fast_p)
        state.slow, slow = ema_step(state.slow, close, slow_p)

        if bar["mark"] == 0:
            # 预热段: 累积 state 但不产信号
            return state, 0

        # fast EMA 必须就绪 (count >= fast_p); slow 在 slow_p 之前返回 0 (NaN 等价);
        # 原版 xp_ema 同形态 (nan_to_num 后 fast>0, slow=0; diff>0 即买)。
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