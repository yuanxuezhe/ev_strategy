from __future__ import annotations
"""ChannelDeviationStrategy: 通道偏离回撤策略 (统一 CPU/GPU; 唯一基类)

================================================================
✅  可改层 (strategies 子包)  ✅
================================================================
混合策略: EMA 通道 + 偏离用 xp 数组算子向量化 (GPU 加速), 锁存 FSM
用 Python 循环 (顺序依赖; cupy 时循环前 .get() 拉回 host)。

compute_signals: 批量向量化 (vectorized 引擎 + sweep)
compute_signals_for_one_bar: 覆写维护 instance FSM (Engine.on_bars / 实盘)

FSM 核心逻辑 (_fsm_step) 只写一次, 两个接口共享 —— 保证语义一致。

参数:
  low1:  下轨极端偏离阈值 (%)  (p0)
  low2:  下轨回撤确认阈值 (%)  (p1)
  high1: 上轨极端偏离阈值 (%)  (p2)
  high2: 上轨回撤确认阈值 (%)  (p3)
  tf1:   EMA 通道周期           (p4)
"""
import numpy as np

from ..indicators import ema_current, ema_push, xp_ema_channel
from .vectorized_base import VectorizedStrategy, register_strategy


# ============ FSM 核心 (只写一次; 批量 + 逐 bar 共享) ============

def _fsm_step(state, cur_ts, low_dev_h, low_dev, high_dev_l, high_dev,
              low1, low2, high1, high2):
    """单 bar FSM 步进; 返回 signal (0/1/-1), 原地改 state。

    state: dict {"low_hit", "high_hit", "lock_ts", "low_acted", "high_acted"}
    语义 (与原 DSL body 逐行一致):
      - 桶切换 (cur_ts != lock_ts) -> 清 low_acted/high_acted
      - low_dev_h < low2 且 low_hit 且未 acted -> BUY, 清 low_hit, 置 low_acted
      - high_dev_l < high2 且 high_hit 且未 acted -> SELL, 清 high_hit, 置 high_acted
      - low_dev > low1 且未 low_acted -> 置 low_hit, 置 low_acted
      - high_dev > high1 且未 high_acted -> 置 high_hit, 置 high_acted
    """
    # 桶切换清锁
    if cur_ts != state["lock_ts"]:
        state["lock_ts"] = cur_ts
        state["low_acted"] = False
        state["high_acted"] = False

    signal = 0
    if state["low_hit"] and low_dev_h < low2 and not state["low_acted"]:
        signal = 1
        state["low_hit"] = False
        state["low_acted"] = True
    elif state["high_hit"] and high_dev_l < high2 and not state["high_acted"]:
        signal = -1
        state["high_hit"] = False
        state["high_acted"] = True

    if low_dev > low1 and not state["low_acted"]:
        state["low_hit"] = True
        state["low_acted"] = True
    if high_dev > high1 and not state["high_acted"]:
        state["high_hit"] = True
        state["high_acted"] = True

    return signal


def _init_fsm_state():
    return {"low_hit": False, "high_hit": False, "lock_ts": 0,
            "low_acted": False, "high_acted": False}


# ============ 策略类 (单继承 VectorizedStrategy) ============

@register_strategy("channel_deviation")
class ChannelDeviationStrategy(VectorizedStrategy):
    """通道偏离回撤策略 (统一 CPU/GPU)

    compute_signals: EMA+偏离向量化 (xp), FSM Python 循环 (_fsm_step)
    compute_signals_for_one_bar: 覆写维护 instance EMA + FSM (逐 bar O(1))
    """

    params_spec = {
        "low1":  {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "low2":  {"default": 1.0, "type": float, "min": 0.0, "max": 100.0},
        "high1": {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "high2": {"default": 0.5, "type": float, "min": 0.0, "max": 100.0},
        "tf1":   {"default": 21,  "type": int,   "min": 2,   "max": 1000},
    }

    def __init__(self, params: dict | None = None, **kwargs):
        super().__init__(params=params, **kwargs)
        # 逐 bar 路径用的 instance state (compute_signals_for_one_bar 用)
        self._fsm = _init_fsm_state()
        self._up_st = (0.0, 0, 0.0)
        self._dw_st = (0.0, 0, 0.0)
        self._has_prev = False
        self._prev_ts = 0
        self._cur_high = 0.0
        self._cur_low = 0.0

    # ---- 批量向量化路径 (vectorized 引擎 / sweep) ----

    def compute_signals(self, xp, bars: dict, params: dict):
        tf1 = int(params["tf1"])
        low1 = float(params["low1"]); low2 = float(params["low2"])
        high1 = float(params["high1"]); high2 = float(params["high2"])

        # 1) 向量化: EMA 通道 + 4 个偏离 (xp 数组算子)
        #    EMA 在桶级 finalized OHLCV 上算 (与 Engine.on_bars 桶 CLOSE 语义对齐:
        #    桶切换时 push 上一桶 high, 然后 ema_current 用本桶 finalized high)
        up, dw = xp_ema_channel(xp, bars["h"], bars["l"], tf1)
        # NaN -> 0 (未就绪时偏离不触发; FSM 会因 low_dev=0 不置位)
        up_v = xp.where(xp.isnan(up), 0.0, up)
        dw_v = xp.where(xp.isnan(dw), 0.0, dw)
        low_dev = (dw_v - bars["l"]) / dw_v * 100.0
        high_dev = (bars["h"] - up_v) / up_v * 100.0
        low_dev_h = (dw_v - bars["h"]) / dw_v * 100.0
        high_dev_l = (bars["l"] - up_v) / up_v * 100.0

        # 2) FSM (Python 循环; cupy 时先拉回 host)
        ts = bars["ts"]; mark = bars["mark"]
        if hasattr(ts, "get"):
            ts = ts.get(); mark = mark.get()
            low_dev_h = low_dev_h.get(); low_dev = low_dev.get()
            high_dev_l = high_dev_l.get(); high_dev = high_dev.get()

        n = len(ts)
        sig = np.zeros(n, dtype=np.int8)
        state = _init_fsm_state()
        for i in range(n):
            if mark[i] == 0:
                continue
            s = _fsm_step(state, int(ts[i]),
                          float(low_dev_h[i]), float(low_dev[i]),
                          float(high_dev_l[i]), float(high_dev[i]),
                          low1, low2, high1, high2)
            sig[i] = s
        return sig

    # ---- 逐 bar 路径 (Engine.on_bars / 实盘; 覆写避免每根 O(n) 重算) ----

    def compute_signals_for_one_bar(self, xp, bar: dict, params: dict) -> int:
        """单 bar: 增量 EMA + _fsm_step (instance FSM 跨桶持续)"""
        tf1 = int(params["tf1"])
        cur_ts = int(bar["ts"])
        cur_high = float(bar["h"])
        cur_low = float(bar["l"])

        # 桶切换 push (旧桶 high/low 闭锁入 EMA)
        if self._has_prev and self._prev_ts != cur_ts:
            us, uc, ue = self._up_st
            us, uc, ue = ema_push(us, uc, ue, self._cur_high, tf1)
            self._up_st = (us, uc, ue)
            ds, dc, de = self._dw_st
            ds, dc, de = ema_push(ds, dc, de, self._cur_low, tf1)
            self._dw_st = (ds, dc, de)

        self._prev_ts = cur_ts
        self._cur_high = cur_high
        self._cur_low = cur_low
        self._has_prev = True

        # 当前通道值 (含 pending)
        us, uc, ue = self._up_st
        ds, dc, de = self._dw_st
        up = ema_current(us, uc, ue, cur_high, tf1)
        dw = ema_current(ds, dc, de, cur_low, tf1)
        if up == 0.0 or dw == 0.0:
            return 0

        low_dev = (dw - cur_low) / dw * 100.0
        high_dev = (cur_high - up) / up * 100.0
        low_dev_h = (dw - cur_high) / dw * 100.0
        high_dev_l = (cur_low - up) / up * 100.0

        return _fsm_step(self._fsm, cur_ts, low_dev_h, low_dev,
                         high_dev_l, high_dev,
                         float(params["low1"]), float(params["low2"]),
                         float(params["high1"]), float(params["high2"]))

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
