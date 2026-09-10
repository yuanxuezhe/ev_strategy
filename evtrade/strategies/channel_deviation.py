"""ChannelDeviationStrategy: 通道偏离回撤策略

EMA 通道 (stateful) + 偏离 (stateless) + 桶级锁存 FSM (stateful)。

参数:
  low1/high1  下/上轨极端偏离阈值 (%)  触发锁存
  low2/high2  下/上轨回撤确认阈值 (%)  触发下单
  tf1         EMA 通道周期
"""
from dataclasses import dataclass, field

from ..indicators import EMAChannelState, ema_channel_step
from ..primitives import fmt, sig_to_side
from .vectorized_base import VectorizedStrategy, register_strategy


# ============ 偏离量 (单桶 OHLCV vs 通道上/下轨) ============

@dataclass
class Deviations:
    """单桶 4 个偏离量 (单位 %); 通道未就绪时全 0"""
    low: float = 0.0      # (DW - L) / DW * 100  桶低 vs 下轨
    high: float = 0.0     # (H - UP) / UP * 100  桶高 vs 上轨
    low_h: float = 0.0    # (DW - H) / DW * 100  桶高回测下轨
    high_l: float = 0.0   # (L - UP) / UP * 100  桶低回测上轨


def _compute_devs(up: float, dw: float, h: float, l: float) -> Deviations:
    """通道上/下轨 + 单桶 OHLCV -> 4 个偏离"""
    if up == 0.0 or dw == 0.0:
        return Deviations()
    return Deviations(
        low=(dw - l) / dw * 100.0,
        high=(h - up) / up * 100.0,
        low_h=(dw - h) / dw * 100.0,
        high_l=(l - up) / up * 100.0,
    )


# ============ FSM 桶级锁存 ============

@dataclass
class DeviationFSM:
    """桶级锁存: 同一桶至多一次 BUY + 一次 SELL"""
    lock_ts: int = 0
    low_hit: bool = False
    high_hit: bool = False
    low_acted: bool = False
    high_acted: bool = False


def _fsm_step(fsm: DeviationFSM, cur_ts: int, devs: Deviations,
              low1: float, low2: float, high1: float, high2: float) -> int:
    """FSM 单步: 桶切换清锁 -> 触发检测/确认下单

    返回: signal ∈ {-1, 0, 1}
    """
    # 桶切换清锁
    if cur_ts != fsm.lock_ts:
        fsm.lock_ts = cur_ts
        fsm.low_acted = False
        fsm.high_acted = False

    signal = 0
    if fsm.low_hit and devs.low_h < low2 and not fsm.low_acted:
        signal = 1
        fsm.low_hit = False
        fsm.low_acted = True
    elif fsm.high_hit and devs.high_l < high2 and not fsm.high_acted:
        signal = -1
        fsm.high_hit = False
        fsm.high_acted = True

    if devs.low > low1 and not fsm.low_acted:
        fsm.low_hit = True
        fsm.low_acted = True
    if devs.high > high1 and not fsm.high_acted:
        fsm.high_hit = True
        fsm.high_acted = True

    return signal


# ============ 策略持久状态 ============

@dataclass
class ChannelDeviationState:
    """策略持久状态: EMA 通道 + FSM + 桶切换检测"""
    ema: EMAChannelState = field(default_factory=EMAChannelState)
    fsm: DeviationFSM = field(default_factory=DeviationFSM)
    prev_ts: int = 0
    cur_high: float = 0.0
    cur_low: float = 0.0
    has_prev: bool = False


# ============ 策略类 ============

@register_strategy("channel_deviation")
class ChannelDeviationStrategy(VectorizedStrategy):
    """通道偏离回撤策略"""

    params_spec = {
        "low1":  {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "low2":  {"default": 1.0, "type": float, "min": 0.0, "max": 100.0},
        "high1": {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "high2": {"default": 0.5, "type": float, "min": 0.0, "max": 100.0},
        "tf1":   {"default": 21,  "type": int,   "min": 2,   "max": 1000},
    }

    def init_state(self, params: dict) -> ChannelDeviationState:
        return ChannelDeviationState()

    def step(self, state: ChannelDeviationState, bar: dict, params: dict
             ) -> tuple[ChannelDeviationState, int]:
        if bar["mark"] == 0:
            return state, 0

        tf1 = int(params["tf1"])
        cur_ts = int(bar["ts"])
        cur_high = float(bar["h"])
        cur_low = float(bar["l"])

        # 桶切换: 旧桶 high/low 闭锁入 EMA
        if state.has_prev and state.prev_ts != cur_ts:
            state.ema, _, _ = ema_channel_step(
                state.ema, state.cur_high, state.cur_low, tf1)

        state.prev_ts = cur_ts
        state.cur_high = cur_high
        state.cur_low = cur_low
        state.has_prev = True

        # 当前通道值 (含 pending)
        state.ema, up, dw = ema_channel_step(state.ema, cur_high, cur_low, tf1)
        devs = _compute_devs(up, dw, cur_high, cur_low)
        sig = _fsm_step(state.fsm, cur_ts, devs,
                        float(params["low1"]), float(params["low2"]),
                        float(params["high1"]), float(params["high2"]))
        # 暴露给 format_signal_line 的可选元数据 (engine 读 self._last_info)
        self._last_info = {"up": up, "dw": dw,
                           "low_dev": devs.low, "high_dev": devs.high}
        return state, sig

    def format_signal_line(self, ts: int, sig: int, info: dict | None = None) -> str:
        info = info or {}
        side = sig_to_side(sig)
        prefix = f"{side} >>xxx> " if side else "             "
        return (f"{prefix}[{ts}] | UP={fmt(info.get('up'))} DW={fmt(info.get('dw'))} | "
                f"low_dev(L/DW)={fmt(info.get('low_dev'))}% "
                f"high_dev(H/UP)={fmt(info.get('high_dev'))}%")
