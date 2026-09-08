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
#
# 持久状态字段 (state_spec 投影): low_hit, high_hit, lock_ts, low_acted,
# high_acted。字段名与 numba jitclass / CUDA device 函数 单名空间对齐 (不
# 再用下划线前缀的 _bucket_ts / _low_acted / _high_acted)。
_CHANNEL_DEVIATION_DSL = """
if ctx.up != ctx.up or ctx.dw != ctx.dw or ctx.up == 0 or ctx.dw == 0:
    return 0
if ctx.cur_ts != ctx.lock_ts:
    ctx.lock_ts = ctx.cur_ts
    ctx.low_acted = False
    ctx.high_acted = False
ctx.low_dev = (ctx.dw - ctx.cur_low) / ctx.dw * 100
ctx.high_dev = (ctx.cur_high - ctx.up) / ctx.up * 100
ctx.low_dev_h = (ctx.dw - ctx.cur_high) / ctx.dw * 100
ctx.high_dev_l = (ctx.cur_low - ctx.up) / ctx.up * 100
# 参数访问: 按 params_spec 顺序 p0=low1, p1=low2, p2=high1, p3=high2
signal = 0
if ctx.low_hit and ctx.low_dev_h < ctx.p1 and not ctx.low_acted:
    signal = 1
    ctx.low_hit = False
    ctx.low_acted = True
elif ctx.high_hit and ctx.high_dev_l < ctx.p3 and not ctx.high_acted:
    signal = -1
    ctx.high_hit = False
    ctx.high_acted = True
if ctx.low_dev > ctx.p0 and not ctx.low_acted:
    ctx.low_hit = True
    ctx.low_acted = True
if ctx.high_dev > ctx.p2 and not ctx.high_acted:
    ctx.high_hit = True
    ctx.high_acted = True
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

    state_spec (DSL 持久状态, framework 投影到 numba jitclass / CUDA device 函数):
      low_hit / high_hit:    是否在当前桶触发极端偏离
      lock_ts:               上一次锁定的桶 ts (桶切换时清零)
      low_acted / high_acted: 本桶内是否已对低/高信号下过单 (避免重复触发)
    """

    params_spec = {
        "low1":  {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "low2":  {"default": 1.0, "type": float, "min": 0.0, "max": 100.0},
        "high1": {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "high2": {"default": 0.5, "type": float, "min": 0.0, "max": 100.0},
    }

    state_spec = {
        "low_hit":    {"type": bool, "default": False},
        "high_hit":   {"type": bool, "default": False},
        "lock_ts":    {"type": int,  "default": 0},
        "low_acted":  {"type": bool, "default": False},
        "high_acted": {"type": bool, "default": False},
    }

    def __init__(self, params: dict | None = None, **kwargs):
        super().__init__(params=params, **kwargs)
        # ctx 跨桶持久 (state_spec 字段跨 bar 持续); 状态字段初值由 make_dsl_ctx
        # 按 state_spec.default 注入, 不再手工初始化 self.<字段>。
        from .dsl import make_dsl_ctx, make_python_runner
        self._ctx = make_dsl_ctx(type(self))
        # runner 走类级缓存 (make_python_runner 内部按 (cls, source_hash) 复用);
        # 避免每次 __init__ 都 exec 一次原 DSL body。
        self._runner = make_python_runner(type(self), "compute_signal")

    def compute_signal(self, ctx):
        """DSL 主逻辑 (channel_deviation)"""
        pass  # DSL 注入

    def check(self, cur, up_or_indicators, dw=None):
        """兼容旧 (cur, up, dw) 与新 (cur, dict) 两种调用

        直接复用 self._ctx (状态字段在 ctx 上跨桶持久, 不再每 bar 重建 +
        手工 copy-back); 与 dsl_check 同源风格: 把每根 bar 的 cur_*/up/dw/pN
        直接挂到 self._ctx 上, 再跑 runner(ctx); 信号状态 (low_hit/lock_ts/
        low_acted/...) 由 runner 直接修改 ctx。
        """
        if isinstance(up_or_indicators, dict):
            up = up_or_indicators.get("up")
            dw_val = up_or_indicators.get("dw")
        else:
            up = up_or_indicators
            dw_val = dw
        if up is None or dw_val is None or up == 0 or dw_val == 0:
            return None, {}

        ctx = self._ctx
        ctx.cur_ts = cur["ts"]
        ctx.cur_open = float(cur.get("open", 0.0))
        ctx.cur_high = float(cur["high"])
        ctx.cur_low = float(cur["low"])
        ctx.cur_close = float(cur["close"])
        ctx.cur_volume = float(cur.get("volume", 0.0))
        ctx.up = up
        ctx.dw = dw_val
        for i, k in enumerate(self.params_spec):
            setattr(ctx, f"p{i}", float(self.params[k]))
        sig_int = int(self._runner(ctx))
        # low_dev* 是 DSL body 里挂在 ctx 上的临时属性, 由 _runner 写入
        signal = {1: "BUY", -1: "SELL"}.get(sig_int)
        info = {"up": up, "dw": dw_val,
                "low_dev": ctx.low_dev, "high_dev": ctx.high_dev,
                "low_dev_h": ctx.low_dev_h, "high_dev_l": ctx.high_dev_l,
                "low_hit_prev": ctx.low_hit,
                "high_hit_prev": ctx.high_hit}
        return signal, info

    def format_signal_line(self, cur, signal, info):
        """verbose 引擎的信号行 (原 Engine.on_bars 内联格式, 原样迁入)

        framework 不假定任何指标字段; channel_deviation 把 up/dw 与偏离值
        都放在 info dict 里, 由本方法自行取用 (hook 签名无 positional 指标)。
        """
        from ..primitives import fmt
        prefix = f"{signal} >>> " if signal else "             "
        return (f"{prefix}[{cur['ts']}] {cur['code']} | O:{cur['open']} H:{cur['high']} "
                f"L:{cur['low']} C:{cur['close']} | vol:{cur['volume']} x{cur['count']} | "
                f"UP={fmt(info.get('up'))} DW={fmt(info.get('dw'))} | "
                f"low_dev(L/DW)={fmt(info.get('low_dev'))}% "
                f"high_dev(H/UP)={fmt(info.get('high_dev'))}% | "
                f"low_dev_h(H/DW)={fmt(info.get('low_dev_h'))}% "
                f"high_dev_l(L/UP)={fmt(info.get('high_dev_l'))}% | "
                f"low_hit_prev={fmt(info.get('low_hit_prev'))} "
                f"high_hit_prev={fmt(info.get('high_hit_prev'))}")

    def get_extra_bucket_columns(self, *, tab: dict, per_bar: dict) -> dict:
        """策略额外列: 4 个偏离百分比 (与 DSL body 同式)

        framework.bucket_table() 只提供 OHLCV + 信号轨迹 (不输出指标);
        per_bar 由 framework 统一打包 (core.kernel.bundle_per_bar / replay),
        策略按需取 per_bar["up"] / ["dw"] / ["h"] / ["l"] 等 per-bar 数组
        计算偏离, 再按 tab['count'] 桶聚合到与 tab['ts'] 等长。
        """
        import numpy as np
        up = per_bar.get("up")
        dw = per_bar.get("dw")
        h = per_bar.get("h")
        l = per_bar.get("l")
        if up is None or dw is None or h is None or l is None:
            return {}
        per_bar_dev = _compute_deviation_columns(up, dw, h, l)
        last_idx = np.cumsum(tab["count"]) - 1
        return {k: v[last_idx] for k, v in per_bar_dev.items()}

    def get_extra_signal_columns(self, *, sig, per_bar: dict) -> dict:
        """策略信号轨迹额外列: EMA 通道 up/dw (per-bar)

        framework --signals-out 默认仅写 (stime, signal); channel_deviation
        额外暴露 per-bar 通道值, 便于离线复盘脚本画图。per_bar 由 framework
        统一打包 (replay_kernel 返回值), 策略按需取值。
        """
        up = per_bar.get("up")
        dw = per_bar.get("dw")
        if up is None or dw is None:
            return {}
        return {"up": up, "dw": dw}


ChannelDeviationStrategy.compute_signal.__doc__ = _CHANNEL_DEVIATION_DSL


def _compute_deviation_columns(up: np.ndarray, dw: np.ndarray,
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
