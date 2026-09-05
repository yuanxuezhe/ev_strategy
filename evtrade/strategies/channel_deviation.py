from __future__ import annotations
"""ChannelDeviationStrategy: 通道偏离回撤策略 (DSL 版)

================================================================
✅  可改层 (strategies 子包)  ✅
================================================================
DSL 版 (走参考引擎)。冻结版在 evtrade.frozen.strategy (kernel 锁定)。

参数通过 params_spec + params dict 传递, 不再靠固定 __init__ 关键字。
"""
from .base import StrategyBase, register_strategy


# DSL 主逻辑 (白名单: 算术/比较/布尔/标量赋值/return)
_CHANNEL_DEVIATION_DSL = """
if ctx.up != ctx.up or ctx.dw != ctx.dw or ctx.up == 0 or ctx.dw == 0:
    return 0
if ctx.cur_ts != ctx._bucket_ts:
    ctx._bucket_ts = ctx.cur_ts
    ctx._low_acted = False
    ctx._high_acted = False
ctx.low_dev = (ctx.dw - ctx.cur_low) / ctx.dw * 100
ctx.high_dev = (ctx.cur_high - ctx.up) / ctx.up * 100
ctx.low_dev_h = (ctx.dw - ctx.cur_high) / ctx.dw * 100
ctx.high_dev_l = (ctx.cur_low - ctx.up) / ctx.up * 100
# 参数访问: 按 params_spec 顺序 p0=low1, p1=low2, p2=high1, p3=high2
if ctx.low_hit and ctx.low_dev_h < ctx.p1 and not ctx._low_acted:
    ctx.low_hit = False
    ctx._low_acted = True
    return 1
if ctx.high_hit and ctx.high_dev_l < ctx.p3 and not ctx._high_acted:
    ctx.high_hit = False
    ctx._high_acted = True
    return -1
if ctx.low_dev > ctx.p0 and not ctx._low_acted:
    ctx.low_hit = True
    ctx._low_acted = True
if ctx.high_dev > ctx.p2 and not ctx._high_acted:
    ctx.high_hit = True
    ctx._high_acted = True
return 0
"""


@register_strategy("channel_deviation")
class ChannelDeviationStrategy(StrategyBase):
    """通道偏离回撤策略 (有状态)

    params (dict 或 kwargs):
      low1:  下轨极端偏离阈值 (%)
      low2:  下轨回撤确认阈值 (%)
      high1: 上轨极端偏离阈值 (%)
      high2: 上轨回撤确认阈值 (%)
    """

    params_spec = {
        "low1":  {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "low2":  {"default": 1.0, "type": float, "min": 0.0, "max": 100.0},
        "high1": {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "high2": {"default": 0.5, "type": float, "min": 0.0, "max": 100.0},
    }

    def __init__(self, params: dict | None = None, **kwargs):
        super().__init__(params=params, **kwargs)
        self.low_hit = False
        self.high_hit = False
        self._bucket_ts = None
        self._low_acted = False
        self._high_acted = False
        # DSL 编译一次
        from .dsl import make_python_runner
        self._runner = make_python_runner(self, "compute_signal")

    def compute_signal(self, ctx):
        """DSL 主逻辑 (channel_deviation)"""
        pass  # DSL 注入

    def check(self, cur, up_or_indicators, dw=None):
        """兼容旧 (cur, up, dw) 与新 (cur, dict) 两种调用"""
        if isinstance(up_or_indicators, dict):
            up = up_or_indicators.get("up")
            dw_val = up_or_indicators.get("dw")
        else:
            up = up_or_indicators
            dw_val = dw
        if up is None or dw_val is None or up == 0 or dw_val == 0:
            return None, {}

        # DSL 字段映射: p0=low1, p1=low2, p2=high1, p3=high2
        # (与 params_spec 顺序一致, 让 DSL 与 GPU/numba 三端一致)
        ctx = _ChannelDevCtx(
            cur_ts=cur["ts"], cur_high=cur["high"], cur_low=cur["low"],
            cur_close=cur["close"],
            up=up, dw=dw_val,
            p0=self.low1, p1=self.low2, p2=self.high1, p3=self.high2,
            low_hit=self.low_hit, high_hit=self.high_hit,
            _bucket_ts=self._bucket_ts,
            _low_acted=self._low_acted,
            _high_acted=self._high_acted)
        sig_int = self._runner(ctx)
        self.low_hit = ctx.low_hit
        self.high_hit = ctx.high_hit
        self._bucket_ts = ctx._bucket_ts
        self._low_acted = ctx._low_acted
        self._high_acted = ctx._high_acted

        signal = {1: "BUY", -1: "SELL"}.get(sig_int)
        info = {"low_dev": ctx.low_dev, "high_dev": ctx.high_dev,
                "low_dev_h": ctx.low_dev_h, "high_dev_l": ctx.high_dev_l,
                "low_hit_prev": self.low_hit, "high_hit_prev": self.high_hit}
        return signal, info


class _ChannelDevCtx:
    """DSL 上下文: 字段顺序与 GPU 通用 kernel 的 p0..p7 寄存器对齐

    p0..p7 是策略参数 (按 params_spec 顺序); pN 之后是状态字段。
    """
    __slots__ = ("cur_ts", "cur_high", "cur_low", "cur_close",
                 "up", "dw", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",
                 "low_hit", "high_hit", "_bucket_ts", "_low_acted", "_high_acted",
                 "low_dev", "high_dev", "low_dev_h", "high_dev_l")

    def __init__(self, cur_ts, cur_high, cur_low, cur_close,
                 up, dw,
                 p0=0.0, p1=0.0, p2=0.0, p3=0.0, p4=0.0, p5=0.0, p6=0.0, p7=0.0,
                 low_hit=False, high_hit=False,
                 _bucket_ts=None, _low_acted=False, _high_acted=False):
        self.cur_ts = cur_ts
        self.cur_high = cur_high
        self.cur_low = cur_low
        self.cur_close = cur_close
        self.up = up
        self.dw = dw
        self.p0, self.p1, self.p2, self.p3 = p0, p1, p2, p3
        self.p4, self.p5, self.p6, self.p7 = p4, p5, p6, p7
        self.low_hit, self.high_hit = low_hit, high_hit
        self._bucket_ts = _bucket_ts
        self._low_acted = _low_acted
        self._high_acted = _high_acted
        self.low_dev = 0.0
        self.high_dev = 0.0
        self.low_dev_h = 0.0
        self.high_dev_l = 0.0


ChannelDeviationStrategy.compute_signal.__doc__ = _CHANNEL_DEVIATION_DSL
