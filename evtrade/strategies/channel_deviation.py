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


# ============ 共享 helpers (去重 compute_signals / _for_one_bar, dedupe-fsm-helpers) ============

def _parse_thresholds(params: dict) -> tuple[float, float, float, float]:
    """(low1, low2, high1, high2) 一次性解析, 避免两处重复写 4 个 float(params[k])"""
    return (float(params["low1"]), float(params["low2"]),
            float(params["high1"]), float(params["high2"]))


def _compute_devs_scalar(up: float, dw: float, h: float, l: float):
    """4 个偏离 (low_dev, high_dev, low_dev_h, high_dev_l) 标量版

    逐 bar 路径用 (compute_signals_for_one_bar); 通道未就绪 (up==0 或 dw==0)
    时返 4 个 0.0, 避免 RuntimeWarning: divide by zero。
    """
    if up == 0.0 or dw == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    return ((dw - l) / dw * 100.0,
            (h - up) / up * 100.0,
            (dw - h) / dw * 100.0,
            (l - up) / up * 100.0)


def _compute_devs_xp(xp, up, dw, h, l):
    """4 个偏离 xp 数组算子版 (批量 compute_signals 用; GPU 加速)

    与原版差异 (顺手修): 原版 (dw_v - l) / dw_v 在 dw_v==0 时产生 inf,
    FSM 把 inf 视为大数会误触发。新版用 safe_dw/safe_up (0->1) 计算,
    最后用 ready mask 强制未就绪处返 0。NaN 由 up_v/dw_v 替换前置掉。
    """
    up_v = xp.where(xp.isnan(up), 0.0, up)
    dw_v = xp.where(xp.isnan(dw), 0.0, dw)
    safe_up = xp.where(up_v == 0, 1.0, up_v)
    safe_dw = xp.where(dw_v == 0, 1.0, dw_v)
    low_dev    = (safe_dw - l)   / safe_dw * 100.0
    high_dev   = (h - safe_up)   / safe_up * 100.0
    low_dev_h  = (safe_dw - h)   / safe_dw * 100.0
    high_dev_l = (l - safe_up)   / safe_up * 100.0
    ready = (up_v != 0) & (dw_v != 0)
    return (xp.where(ready, low_dev,    0.0),
            xp.where(ready, high_dev,   0.0),
            xp.where(ready, low_dev_h,  0.0),
            xp.where(ready, high_dev_l, 0.0))


def _run_fsm(state, ts, low_dev_h, low_dev, high_dev_l, high_dev,
             low1, low2, high1, high2) -> int:
    """薄包装 _fsm_step; 把传参顺序定死避免两处调用错位"""
    return _fsm_step(state, ts,
                     low_dev_h, low_dev, high_dev_l, high_dev,
                     low1, low2, high1, high2)


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
        low1, low2, high1, high2 = _parse_thresholds(params)

        # 1) 向量化: EMA 通道 + 4 个偏离 (xp 数组算子; GPU 加速)
        #    EMA 在桶级 finalized OHLCV 上算 (与 Engine.on_bars 桶 CLOSE 语义对齐:
        #    桶切换时 push 上一桶 high, 然后 ema_current 用本桶 finalized high)
        up, dw = xp_ema_channel(xp, bars["h"], bars["l"], tf1)
        # NaN -> 0 + 通道未就绪 (up==0/dw==0) -> 0 由 _compute_devs_xp 处理
        low_dev, high_dev, low_dev_h, high_dev_l = _compute_devs_xp(
            xp, up, dw, bars["h"], bars["l"])

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
            sig[i] = _run_fsm(state, int(ts[i]),
                              float(low_dev_h[i]), float(low_dev[i]),
                              float(high_dev_l[i]), float(high_dev[i]),
                              low1, low2, high1, high2)
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

        # 当前通道值 (含 pending); 通道未就绪由 _compute_devs_scalar 内部短路
        us, uc, ue = self._up_st
        ds, dc, de = self._dw_st
        up = ema_current(us, uc, ue, cur_high, tf1)
        dw = ema_current(ds, dc, de, cur_low, tf1)

        low_dev, high_dev, low_dev_h, high_dev_l = _compute_devs_scalar(
            up, dw, cur_high, cur_low)
        if up == 0.0 or dw == 0.0:
            return 0  # 通道未就绪: 偏离全 0, FSM 不触发

        return _run_fsm(self._fsm, cur_ts,
                        low_dev_h, low_dev, high_dev_l, high_dev,
                        *_parse_thresholds(params))

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
