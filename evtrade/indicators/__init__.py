from __future__ import annotations
"""indicators 子包: 纯 ndarray / 纯 Python 指标库 (复盘 / compute_signals / Engine 路径)

公开 API:
  === 纯 ndarray 版 (批量, 复盘 / jupyter 友好) ===
  ema(values, p)                   EMA 序列 (前 p-1 为 NaN)
  ema_channel(highs, lows, p)      上轨 + 下轨
  true_range(highs, lows, closes)
  atr(highs, lows, closes, p, ema=False)  Wilder ATR
  rsi(closes, p)                   Wilder RSI
  sma(values, p)                   简单移动平均
  bollinger(closes, p, k)          (middle, upper, lower)

  === xp 版 (compute_signals 主体调用) ===
  xp_ema(xp, values, p)            xp 兼容 EMA 序列 (前 p-1 为 NaN)
  xp_ema_channel(xp, highs, lows, p) xp 兼容 EMA 通道

  === 纯 Python 增量版 (Engine.on_bars / compute_signals_for_one_bar 路径) ===
  ema_push(s_sum, s_count, s_ema, value, p) -> (sum, count, ema)
  ema_current(s_sum, s_count, s_ema, pending, p) -> float
  ema_channel_push(us, uc, ue, ds, dc, de, h, l, p) -> (us, uc, ue, ds, dc, de)
  ema_channel_current(us, uc, ue, ds, dc, de, h, l, p) -> (up, dw)
  atr_push(state, h, l, c, p) -> state           (Wilder 平滑)
  atr_current(state, h, l, c, p) -> float
  rsi_push(state, close, p) -> state             (Wilder 平滑)
  rsi_current(state, close, p) -> float
  sma_push(state, value) -> state
  sma_current(state, pending, p) -> float
  boll_push(state, value) -> state
  boll_current(state, pending, p, k) -> (mid, upper, lower)

策略在 compute_signals(xp, bars, params) body 内用 xp 版 (xp_ema 等)；
策略在覆写 compute_signals_for_one_bar 内用 Python 增量版 (ema_push 等)；
DSL docstring / @njit / CUDA __device__ 已下线 (2026-09 unify-strategy-contract)。

state 形状说明:
  ema: (sum, count, ema); 详见 evtrade/indicators/ema.py
  atr: (sum, count, prev_close, atr)
  rsi: (sum_g, sum_l, avg_g, avg_l, count, prev_close)
  sma: (sum, count)
  boll: (sum, sumsq, count)
"""
from .ema import (
    ema, ema_channel,
    xp_ema, xp_ema_channel,
    ema_push, ema_current, ema_channel_push, ema_channel_current,
)
from .atr import (
    true_range, atr,
    atr_push, atr_current,
)
from .rsi import (
    rsi,
    rsi_push, rsi_current,
)
from .boll import (
    sma, bollinger,
    sma_push, sma_current, boll_push, boll_current,
)

__all__ = [
    # 纯 ndarray 版
    "ema", "ema_channel",
    "true_range", "atr",
    "rsi", "sma", "bollinger",
    # xp 版
    "xp_ema", "xp_ema_channel",
    # 纯 Python 增量版
    "ema_push", "ema_current", "ema_channel_push", "ema_channel_current",
    "atr_push", "atr_current",
    "rsi_push", "rsi_current",
    "sma_push", "sma_current", "boll_push", "boll_current",
]
