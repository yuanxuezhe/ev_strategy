from __future__ import annotations
"""indicators 子包: 纯函数指标库 (复盘 / jupyter 友好)

公开 API:
  ema(values, p)            EMA 序列 (前 p-1 为 None)
  ema_channel(highs, lows, p)  上轨 + 下轨
  true_range(highs, lows, closes)
  atr(highs, lows, closes, p, ema=False)  Wilder ATR
  rsi(closes, p)            Wilder RSI
  sma(values, p)            简单移动平均
  bollinger(closes, p, k)   (middle, upper, lower)

注意:
  增量状态机版 (热路径) 在 evtrade.indicators.IncrementalEMA /
  EMAChannel —— Kernel 内核使用, 走 numba njit。
  纯函数版只用于复盘 / 单元测试 / 离线分析, 不进 numba 流式内核。
"""
from .ema import ema, ema_channel
from .atr import true_range, atr
from .rsi import rsi
from .boll import sma, bollinger

__all__ = [
    "ema", "ema_channel",
    "true_range", "atr",
    "rsi", "sma", "bollinger",
]
