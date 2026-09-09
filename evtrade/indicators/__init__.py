from __future__ import annotations
"""indicators 子包: 纯函数 + @njit 增量指标库 (复盘 / DSL 三端可调)

公开 API:
  === 纯函数版 (批量, 复盘 / jupyter 友好) ===
  ema(values, p)                   EMA 序列 (前 p-1 为 None)
  ema_channel(highs, lows, p)      上轨 + 下轨
  true_range(highs, lows, closes)
  atr(highs, lows, closes, p, ema=False)  Wilder ATR
  rsi(closes, p)                   Wilder RSI
  sma(values, p)                   简单移动平均
  bollinger(closes, p, k)          (middle, upper, lower)

  === @njit 增量版 (DSL 三端可调: Python exec / numba @njit / CUDA __device__) ===
  ema_push(state, value, p) -> state
  ema_current(state, pending, p) -> float
  ema_channel_push(up_st, dw_st, h, l, p) -> (up_st, dw_st)
  ema_channel_current(up_st, dw_st, h, l, p) -> (up, dw)
  atr_push(state, h, l, c, p) -> state           (Wilder 平滑)
  atr_current(state, h, l, p) -> float
  rsi_push(state, close, p) -> state             (Wilder 平滑)
  rsi_current(state, close, p) -> float
  sma_push(state, value, p) -> state
  sma_current(state, pending, p) -> float
  boll_push(state, value, p) -> state
  boll_current(state, value, p) -> (mid, upper, lower)

DSL body 通过白名单内的 `ema_channel_push(...)` 等调; 三端渲染由
strategies/dsl.py::render_numba_state_body / render_cuda_device_function
负责 (Python/numba 端 import, CUDA 端内联 __device__ 源码)。

state 形状说明:
  ema: (sum: float, count: int, ema: float); 详见 evtrade/indicators/ema.py
  atr: (sum: float, count: int, prev_close: float, atr: float)
  rsi: (avg_gain: float, avg_loss: float, count: int, prev_close: float)
  sma: (sum: float, count: int)
  boll: (sum: float, sumsq: float, count: int, mid: float, upper: float, lower: float)
"""
from .ema import (
    ema, ema_channel,
    ema_push, ema_current, ema_channel_push, ema_channel_current,
    CUDA_DEVICE_EMA_PUSH, CUDA_DEVICE_EMA_CHANNEL_PUSH, CUDA_DEVICE_EMA_CURRENT,
)
from .atr import (
    true_range, atr,
    atr_push, atr_current,
    CUDA_DEVICE_ATR_PUSH, CUDA_DEVICE_ATR_CURRENT,
)
from .rsi import (
    rsi,
    rsi_push, rsi_current,
    CUDA_DEVICE_RSI_PUSH, CUDA_DEVICE_RSI_CURRENT,
)
from .boll import (
    sma, bollinger,
    sma_push, sma_current, boll_push, boll_current,
    CUDA_DEVICE_SMA_PUSH, CUDA_DEVICE_SMA_CURRENT,
    CUDA_DEVICE_BOLL_PUSH, CUDA_DEVICE_BOLL_CURRENT,
)

__all__ = [
    # 纯函数版
    "ema", "ema_channel",
    "true_range", "atr",
    "rsi", "sma", "bollinger",
    # @njit 增量版 (DSL 三端可调)
    "ema_push", "ema_current", "ema_channel_push", "ema_channel_current",
    "atr_push", "atr_current",
    "rsi_push", "rsi_current",
    "sma_push", "sma_current", "boll_push", "boll_current",
    # CUDA __device__ 源码 (DSL CUDA 渲染器内联)
    "CUDA_DEVICE_EMA_PUSH", "CUDA_DEVICE_EMA_CHANNEL_PUSH", "CUDA_DEVICE_EMA_CURRENT",
    "CUDA_DEVICE_ATR_PUSH", "CUDA_DEVICE_ATR_CURRENT",
    "CUDA_DEVICE_RSI_PUSH", "CUDA_DEVICE_RSI_CURRENT",
    "CUDA_DEVICE_SMA_PUSH", "CUDA_DEVICE_SMA_CURRENT",
    "CUDA_DEVICE_BOLL_PUSH", "CUDA_DEVICE_BOLL_CURRENT",
]
