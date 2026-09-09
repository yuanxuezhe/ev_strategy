from __future__ import annotations
"""indicators 子包: 纯 ndarray / 纯 Python 指标库 (strategy-step-only, 2026-09-10)

公开 API:
  === 纯 ndarray 版 (批量, 复盘 / jupyter 友好) ===
  ema(values, p)                   EMA 序列 (前 p-1 为 NaN)
  ema_channel(highs, lows, p)      上轨 + 下轨
  true_range(highs, lows, closes)
  atr(highs, lows, closes, p, ema=False)  Wilder ATR
  rsi(closes, p)                   Wilder RSI
  sma(values, p)                   简单移动平均
  bollinger(closes, p, k)          (middle, upper, lower)

  === xp 版 (engine fast-path 调用; 策略不直接用) ===
  xp_ema(xp, values, p)            xp 兼容 EMA 序列
  xp_ema_channel(xp, highs, lows, p) xp 兼容 EMA 通道

  === step 增量版 (策略 step() 调, 2026-09-10 统一) ===
  EMAState / EMAChannelState / ATRState / RSIState / SMAState / BollState  dataclass
  ema_step(state, value, p) -> (state, ema)
  ema_channel_step(state, h, l, p) -> (state, up, dw)
  atr_step(state, h, l, c, p) -> (state, atr)
  rsi_step(state, close, p) -> (state, rsi)
  sma_step(state, value, p) -> (state, sma)
  boll_step(state, value, p, k) -> (state, mid, upper, lower)

  === Deprecated shim (过渡期; 后续 change 删除) ===
  ema_push / ema_current / ema_channel_push / ema_channel_current
  atr_push / atr_current
  rsi_push / rsi_current
  sma_push / sma_current / boll_push / boll_current

策略在 step(state, bar, params) body 内调 ema_step 等 (stateful, 逐桶);
引擎 fast-path 用 xp 版一次性算全序列 (stateful step 之外, 性能优化)。

DSL docstring / @njit / CUDA __device__ 已下线 (2026-09 unify-strategy-contract)。
"""
from .ema import (
    ema, ema_channel,
    xp_ema, xp_ema_channel,
    # step API (strategy-step-only)
    EMAState, EMAChannelState,
    ema_step, ema_channel_step,
    # deprecated shim
    ema_push, ema_current, ema_channel_push, ema_channel_current,
)
from .atr import (
    true_range, atr,
    # step API
    ATRState, atr_step,
    # deprecated shim
    atr_push, atr_current,
)
from .rsi import (
    rsi,
    # step API
    RSIState, rsi_step,
    # deprecated shim
    rsi_push, rsi_current,
)
from .boll import (
    sma, bollinger,
    # step API
    SMAState, BollState, sma_step, boll_step,
    # deprecated shim
    sma_push, sma_current, boll_push, boll_current,
)

__all__ = [
    # 纯 ndarray 版
    "ema", "ema_channel",
    "true_range", "atr",
    "rsi", "sma", "bollinger",
    # xp 版
    "xp_ema", "xp_ema_channel",
    # step dataclass
    "EMAState", "EMAChannelState",
    "ATRState", "RSIState", "SMAState", "BollState",
    # step API
    "ema_step", "ema_channel_step",
    "atr_step", "rsi_step", "sma_step", "boll_step",
    # 纯 Python 增量版 (deprecated shim)
    "ema_push", "ema_current", "ema_channel_push", "ema_channel_current",
    "atr_push", "atr_current",
    "rsi_push", "rsi_current",
    "sma_push", "sma_current", "boll_push", "boll_current",
]