"""indicators 子包: 纯 ndarray / 纯 Python 指标库

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

  === torch 版 ===
  xp_ema_torch(values, p)          torch.Tensor -> torch.Tensor
  xp_ema_channel_torch(highs, lows, p) -> (up, dw)

  === step 增量版 (策略 step() 调) ===
  EMAState / EMAChannelState / ATRState / RSIState / SMAState / BollState  dataclass
  ema_step(state, value, p) -> (state, ema)
  ema_channel_step(state, h, l, p) -> (state, up, dw)
  atr_step(state, h, l, c, p) -> (state, atr)
  rsi_step(state, close, p) -> (state, rsi)
  sma_step(state, value, p) -> (state, sma)
  boll_step(state, value, p, k) -> (state, mid, upper, lower)

策略在 step(state, bar, params) body 内调 ema_step 等 (stateful, 逐桶);
引擎 fast-path 用 xp 版一次性算全序列 (stateful step 之外, 性能优化)。
"""
from .ema import (
    ema, ema_channel,
    xp_ema, xp_ema_channel,
    xp_ema_torch, xp_ema_channel_torch,
    EMAState, EMAChannelState,
    ema_step, ema_channel_step,
)
from .atr import (
    true_range, atr,
    ATRState, atr_step,
)
from .rsi import (
    rsi,
    RSIState, rsi_step,
)
from .boll import (
    sma, bollinger,
    SMAState, BollState, sma_step, boll_step,
)

__all__ = [
    "ema", "ema_channel",
    "true_range", "atr",
    "rsi", "sma", "bollinger",
    "xp_ema", "xp_ema_channel",
    "xp_ema_torch", "xp_ema_channel_torch",
    "EMAState", "EMAChannelState",
    "ATRState", "RSIState", "SMAState", "BollState",
    "ema_step", "ema_channel_step",
    "atr_step", "rsi_step", "sma_step", "boll_step",
]