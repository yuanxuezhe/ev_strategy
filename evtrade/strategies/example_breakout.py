from __future__ import annotations
"""示例策略: 区间突破 (示意用, 仅参考引擎)

演示 params_spec + dict 参数通用模式。
"""
from .base import StrategyBase, register_strategy


@register_strategy("breakout")
class BreakoutStrategy(StrategyBase):
    """N 根 1m bar 区间突破

    params (dict):
      lookback:      回看窗口 (bar 数)
      breakout_pct:  突破百分比阈值 (0.001 = 0.1%)
    """

    params_spec = {
        "lookback":     {"default": 20,    "type": int,   "min": 2,    "max": 1000},
        "breakout_pct": {"default": 0.001, "type": float, "min": 0.0,  "max": 0.1},
    }

    def __init__(self, params: dict | None = None, **kwargs):
        super().__init__(params=params, **kwargs)
        self._bucket_ts = None
        self._acted = False
        self._highs: list[float] = []
        self._lows: list[float] = []

    def _push(self, h: float, l: float):
        self._highs.append(h)
        self._lows.append(l)
        if len(self._highs) > self.lookback:
            self._highs.pop(0)
            self._lows.pop(0)

    def check(self, cur, up_or_indicators, dw=None):
        if cur["ts"] != self._bucket_ts:
            self._bucket_ts = cur["ts"]
            self._acted = False

        self._push(cur["high"], cur["low"])
        if len(self._highs) < self.lookback:
            return None, {}

        if self._acted:
            return None, {}

        hi = max(self._highs)
        lo = min(self._lows)
        c = cur["close"]
        info = {"hi": hi, "lo": lo, "close": c}

        if c > hi * (1 + self.breakout_pct):
            self._acted = True
            return "BUY", info
        if c < lo * (1 - self.breakout_pct):
            self._acted = True
            return "SELL", info
        return None, info


# 注册 sweep 网格允许的参数名 (框架层 GRID_KEYS 只放引擎级, 策略参数自注册)
from ..core.sweep import register_grid_keys
register_grid_keys(set(BreakoutStrategy.params_spec.keys()))
