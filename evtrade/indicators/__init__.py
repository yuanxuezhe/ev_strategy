"""indicators 子包: EMA 指标 (仅 EMA 一种指标, 两形态)

公开 API:
  === step 增量版 (策略 step() 逐桶调用) ===
  EMAState / EMAChannelState       dataclass state
  ema_step(state, value, p) -> (state, ema)
  ema_channel_step(state, h, l, p) -> (state, up, dw)

  === numpy 批量参考版 (reconcile 参考实现 / 测试用) ===
  ema(values, p)                   EMA 序列 (前 p-1 为 NaN)
  ema_channel(highs, lows, p)      上轨 + 下轨

策略在 step(state, bar, params) body 内调 ema_step / ema_channel_step;
引擎不预计算任何指标。
"""
from .ema import (
    EMAChannelState, EMAState,
    ema, ema_channel,
    ema_channel_step, ema_step,
)

__all__ = [
    "ema", "ema_channel",
    "EMAState", "EMAChannelState",
    "ema_step", "ema_channel_step",
]
