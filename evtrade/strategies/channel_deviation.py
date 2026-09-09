from __future__ import annotations
"""ChannelDeviationStrategy: 通道偏离回撤策略

================================================================
✅  可改层 (strategies 子包)  ✅
================================================================
混合策略: EMA 通道 (stateful, step) + 偏离 (stateless, 算) + 锁存 FSM (stateful, step)。

唯一方法 step(state, bar, params) -> (state, sig):
  - state 由 engine 持有传入, 策略无 instance attr
  - bar 是单桶 OHLCV + mark; mark=0 (预热) 直接返 (state, 0)
  - 算法: ema_channel_step 算 up/dw + 4 偏离 + _fsm_step 锁存

参数:
  low1:  下轨极端偏离阈值 (%)  (p0)
  low2:  下轨回撤确认阈值 (%)  (p1)
  high1: 上轨极端偏离阈值 (%)  (p2)
  high2: 上轨回撤确认阈值 (%)  (p3)
  tf1:   EMA 通道周期           (p4)
"""
from dataclasses import dataclass, field

from ..indicators import EMAChannelState, ema_channel_step
from .vectorized_base import VectorizedStrategy, register_strategy


# ============ 状态 dataclass ============

@dataclass
class ChannelDeviationState:
    """策略持久状态: EMA 通道 + FSM 锁存

    engine 在 strategy 实例化时调 init_state() 拿初值, 之后每次 step 调用传入传出。
    """
    ema: EMAChannelState = field(default_factory=EMAChannelState)
    fsm: dict = field(default_factory=lambda: {
        "low_hit": False, "high_hit": False, "lock_ts": 0,
        "low_acted": False, "high_acted": False})
    prev_ts: int = 0
    cur_high: float = 0.0
    cur_low: float = 0.0
    has_prev: bool = False


# ============ 算法 ============

def _fsm_step(state_fsm: dict, cur_ts: int, low_dev_h, low_dev,
              high_dev_l, high_dev, low1, low2, high1, high2) -> int:
    """FSM 单步: 原地改 state_fsm, 返回 signal (0/1/-1)

    语义 (与原 DSL body 逐行一致):
      - 桶切换 (cur_ts != lock_ts) -> 清 low_acted/high_acted
      - low_dev_h < low2 且 low_hit 且未 acted -> BUY, 清 low_hit, 置 low_acted
      - high_dev_l < high2 且 high_hit 且未 acted -> SELL, 清 high_hit, 置 high_acted
      - low_dev > low1 且未 low_acted -> 置 low_hit, 置 low_acted
      - high_dev > high1 且未 high_acted -> 置 high_hit, 置 high_acted
    """
    # 桶切换清锁
    if cur_ts != state_fsm["lock_ts"]:
        state_fsm["lock_ts"] = cur_ts
        state_fsm["low_acted"] = False
        state_fsm["high_acted"] = False

    signal = 0
    if state_fsm["low_hit"] and low_dev_h < low2 and not state_fsm["low_acted"]:
        signal = 1
        state_fsm["low_hit"] = False
        state_fsm["low_acted"] = True
    elif state_fsm["high_hit"] and high_dev_l < high2 and not state_fsm["high_acted"]:
        signal = -1
        state_fsm["high_hit"] = False
        state_fsm["high_acted"] = True

    if low_dev > low1 and not state_fsm["low_acted"]:
        state_fsm["low_hit"] = True
        state_fsm["low_acted"] = True
    if high_dev > high1 and not state_fsm["high_acted"]:
        state_fsm["high_hit"] = True
        state_fsm["high_acted"] = True

    return signal


def _compute_devs(up: float, dw: float, h: float, l: float):
    """4 个偏离 (low_dev, high_dev, low_dev_h, high_dev_l); 通道未就绪返 0"""
    if up == 0.0 or dw == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    return ((dw - l) / dw * 100.0,
            (h - up) / up * 100.0,
            (dw - h) / dw * 100.0,
            (l - up) / up * 100.0)


# ============ 策略类 (单继承 VectorizedStrategy) ============

@register_strategy("channel_deviation")
class ChannelDeviationStrategy(VectorizedStrategy):
    """通道偏离回撤策略 (strategy-step-only, 2026-09-10)

    唯一抽象 step(state, bar, params) -> (state, sig); 无 instance state。
    """

    params_spec = {
        "low1":  {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "low2":  {"default": 1.0, "type": float, "min": 0.0, "max": 100.0},
        "high1": {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "high2": {"default": 0.5, "type": float, "min": 0.0, "max": 100.0},
        "tf1":   {"default": 21,  "type": int,   "min": 2,   "max": 1000},
    }

    def init_state(self, params: dict) -> ChannelDeviationState:
        """engine 调一次, 返回 state 初值"""
        return ChannelDeviationState()

    def step(self, state: ChannelDeviationState, bar: dict, params: dict
             ) -> tuple[ChannelDeviationState, int]:
        """单步: state + 单桶 bar -> (new_state, sig)

        bar = {"ts","o","h","l","c","v","mark"}
        预热段 (mark==0) 直接返 (state, 0)。
        """
        if bar["mark"] == 0:
            return state, 0

        tf1 = int(params["tf1"])
        low1, low2, high1, high2 = (float(params[k]) for k in
                                    ("low1", "low2", "high1", "high2"))
        cur_ts = int(bar["ts"])
        cur_high = float(bar["h"])
        cur_low = float(bar["l"])

        # 桶切换 push (旧桶 high/low 闭锁入 EMA)
        if state.has_prev and state.prev_ts != cur_ts:
            state.ema, _, _ = ema_channel_step(
                state.ema, state.cur_high, state.cur_low, tf1)

        state.prev_ts = cur_ts
        state.cur_high = cur_high
        state.cur_low = cur_low
        state.has_prev = True

        # 当前通道值 (含 pending); 通道未就绪返 0 由 _compute_devs 内部短路
        state.ema, up, dw = ema_channel_step(state.ema, cur_high, cur_low, tf1)
        low_dev, high_dev, low_dev_h, high_dev_l = _compute_devs(
            up, dw, cur_high, cur_low)
        sig = _fsm_step(state.fsm, cur_ts,
                        low_dev_h, low_dev, high_dev_l, high_dev,
                        low1, low2, high1, high2)
        return state, sig

    # ---- 展示 hook ----

    def format_signal_line(self, ts: int, sig: int, info: dict | None = None) -> str:
        """自定义信号行打印; info 来自 Engine.on_bars 累积 (up/dw/dev)"""
        from ..primitives import fmt
        info = info or {}
        side = {1: "BUY", -1: "SELL"}.get(sig, "")
        prefix = f"{side} >>> " if side else "             "
        return (f"{prefix}[{ts}] | UP={fmt(info.get('up'))} DW={fmt(info.get('dw'))} | "
                f"low_dev(L/DW)={fmt(info.get('low_dev'))}% "
                f"high_dev(H/UP)={fmt(info.get('high_dev'))}%")