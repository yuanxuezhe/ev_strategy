from __future__ import annotations
"""ChannelDeviationStrategy: 通道偏离回撤策略 (DSL 版)

================================================================
✅  可改层 (strategies 子包)  ✅
================================================================
DSL 版 (走参考引擎 + numba kernel + CUDA, 三端同源)。
唯一 ChannelDeviationStrategy; frozen/strategy.py 已删除 (2026-09 重构)。

参数通过 params_spec + params dict 传递, 不再靠固定 __init__ 关键字。
"""
from .base import StrategyBase, register_strategy


# DSL 主逻辑 (白名单: 算术/比较/布尔/标量赋值/return)
# 信号只赋值不提前 return, 让随后的极端偏离锁存在本桶仍生效。
# 三端 (Python/numba/CUDA) 同源同语义; 公式逐位锁定 (test_differential 72 项
# bitwise 一致, 与原 frozen/strategy.py 历史行为一致, 后者已并入本文件)。
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
signal = 0
if ctx.low_hit and ctx.low_dev_h < ctx.p1 and not ctx._low_acted:
    signal = 1
    ctx.low_hit = False
    ctx._low_acted = True
elif ctx.high_hit and ctx.high_dev_l < ctx.p3 and not ctx._high_acted:
    signal = -1
    ctx.high_hit = False
    ctx._high_acted = True
if ctx.low_dev > ctx.p0 and not ctx._low_acted:
    ctx.low_hit = True
    ctx._low_acted = True
if ctx.high_dev > ctx.p2 and not ctx._high_acted:
    ctx.high_hit = True
    ctx._high_acted = True
return signal
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
        # runner 走类级缓存 (make_python_runner 内部按 (cls, source_hash) 复用);
        # 避免每次 __init__ 都 exec 一次原 DSL body。
        from .dsl import make_python_runner
        self._runner = make_python_runner(type(self), "compute_signal")

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

    def format_signal_line(self, cur, up, dw, signal, info):
        """verbose 引擎的信号行 (原 Engine.on_bars 内联格式, 原样迁入)"""
        from ..primitives import fmt
        prefix = f"{signal} >>> " if signal else "             "
        return (f"{prefix}[{cur['ts']}] {cur['code']} | O:{cur['open']} H:{cur['high']} "
                f"L:{cur['low']} C:{cur['close']} | vol:{cur['volume']} x{cur['count']} | "
                f"UP={fmt(up)} DW={fmt(dw)} | "
                f"low_dev(L/DW)={fmt(info.get('low_dev'))}% "
                f"high_dev(H/UP)={fmt(info.get('high_dev'))}% | "
                f"low_dev_h(H/DW)={fmt(info.get('low_dev_h'))}% "
                f"high_dev_l(L/UP)={fmt(info.get('high_dev_l'))}% | "
                f"low_hit_prev={fmt(info.get('low_hit_prev'))} "
                f"high_hit_prev={fmt(info.get('high_hit_prev'))}")


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


def compute_deviation_columns(up: np.ndarray, dw: np.ndarray,
                              h: np.ndarray, l: np.ndarray) -> dict[str, np.ndarray]:
    """四个偏离值 (策略层公式, 与 DSL body 同式)

    输入: 全轨迹或桶级数组 (等长)
    输出: dict 含 low_dev / high_dev / low_dev_h / high_dev_l, 长度同输入。
          up/dw 无效 (NaN 或 0) 处偏离值为 NaN。

    这是策略专属的"指标计算"函数 (channel_deviation 的语义: 通道上下轨对
    当根 bar high/low 的偏离百分比), 不应放在 framework.bucket_table 里;
    调用方 (CLI 展示 / 复盘工具 / 测试) 需要时自行调用本函数。
    """
    import numpy as np
    valid = np.isfinite(up) & np.isfinite(dw) & (up != 0.0) & (dw != 0.0)
    upv = np.where(valid, up, np.nan)
    dwv = np.where(valid, dw, np.nan)
    return {
        "low_dev":    (dwv - l) / dwv * 100.0,
        "high_dev":   (h - upv) / upv * 100.0,
        "low_dev_h":  (dwv - h) / dwv * 100.0,
        "high_dev_l": (l - upv) / upv * 100.0,
    }


# 注册 sweep 网格允许的参数名 (框架层 GRID_KEYS 只放引擎级, 策略参数自注册)
from ..core.sweep import register_grid_keys
register_grid_keys(set(ChannelDeviationStrategy.params_spec.keys()))
