from __future__ import annotations
"""示例策略: 轨道触发 (DSL 三端同源的最小完整示例)

================================================================
✅  可改层 (strategies 子包)  ✅  —— 新策略抄这个文件改三处
================================================================
与 example_breakout (仅参考引擎) 的区别: 主逻辑写在 compute_signal 的
docstring (DSL) 里, 三端自动可用:
  1. Python 参考引擎: dsl_check + DSLCtx (本文件 check 的两行样板)
  2. numba 内核:      core/kernel_dsl.dsl_kernel("dev_trigger") 渲染特化
  3. CUDA 内核:       gpu.cuda_sweep_window_generic(..., strategy_name=...)

策略语义 (每桶至多一次操作, 桶切换解锁):
  - 低点触及下轨下方 p0% -> BUY (ctx.p0 = params_spec 第 1 个参数 entry_dev)
  - 高点触及上轨上方 p0% -> SELL
参数访问一律用 ctx.pN (按 params_spec 声明顺序), 与内核 p0..p15 对齐。

2026-09: framework 完全解耦 EMA — 策略自维护 EMA 通道 (DSL body 内调
evtrade.indicators.ema.ema_current); check 签名统一为 (cur, indicators=None),
EMA state 由 kernel.step 在桶切换时 push 进 ctx.up_sum/up_count/up_ema (base
state, 任何策略都自带), 与原 _sync_ema 行为逐位一致。
"""
from .base import StrategyBase, register_strategy
from .dsl import make_dsl_ctx, make_python_runner


@register_strategy("dev_trigger")
class DevTriggerStrategy(StrategyBase):
    """轨道触发策略 (有状态, DSL 版)

    params (dict 或 kwargs):
      entry_dev: 触发深度 (%): 低点 < 下轨×(1-p0/100) 买 / 高点 > 上轨×(1+p0/100) 卖
      tf1:       EMA 通道周期 (策略私有; 通道轨由 DSL body 自维护)
    """

    params_spec = {
        "entry_dev": {"default": 0.8, "type": float, "min": 0.0, "max": 50.0},
        "tf1":       {"default": 21,  "type": int,   "min": 2,   "max": 1000},
    }

    state_spec = {
        "lock_ts":    {"type": int,  "default": 0},
        "low_acted":  {"type": bool, "default": False},
        "high_acted": {"type": bool, "default": False},
    }

    def __init__(self, params: dict | None = None, **kwargs):
        super().__init__(params=params, **kwargs)
        self._ctx = make_dsl_ctx(type(self))   # 状态字段跨桶持久
        self._runner = make_python_runner(type(self), "compute_signal")

    def compute_signal(self, ctx):
        """DSL 主逻辑 (dev_trigger)"""
        pass  # DSL 注入

    def check(self, cur, indicators=None):
        """统一签名: check (cur, indicators=None) -> (signal|None, info)

        framework 永远传 indicators=None (Engine.on_bars 签名, 见 09 号文档 §2);
        DSL body 第一段自维护 EMA 通道 (调 evtrade.indicators.ema.ema_current),
        framework 在桶切换时 push 闭合桶 high/low 进 ctx.up_sum/up_count/up_ema
        (与原 _sync_ema 等价)。
        """
        ctx = self._ctx
        ctx.cur_ts = cur["ts"]
        ctx.cur_high = float(cur["high"])
        ctx.cur_low = float(cur["low"])
        ctx.cur_close = float(cur["close"])
        ctx.cur_open = float(cur.get("open", 0.0))
        ctx.cur_volume = float(cur.get("volume", 0.0))
        for i, k in enumerate(self.params_spec):
            setattr(ctx, f"p{i}", float(self.params[k]))
        sig_int = int(self._runner(ctx))
        signal = {1: "BUY", -1: "SELL"}.get(sig_int)
        return signal, {}


DevTriggerStrategy.compute_signal.__doc__ = """\
# === 第一段: 通道值 (framework kernel 桶切换时 push; DSL 只读) ===
ctx.up = ema_current(ctx.up_sum, ctx.up_count, ctx.up_ema, ctx.cur_high, ctx.p1)
ctx.dw = ema_current(ctx.dw_sum, ctx.dw_count, ctx.dw_ema, ctx.cur_low, ctx.p1)
# === 第二段: 桶切换清锁 + 触发 ===
if ctx.up == 0 or ctx.dw == 0:
    return 0
if ctx.cur_ts != ctx.lock_ts:
    ctx.lock_ts = ctx.cur_ts
    ctx.low_acted = False
    ctx.high_acted = False
if ctx.cur_low < ctx.dw * (1 - ctx.p0 / 100) and not ctx.low_acted:
    ctx.low_acted = True
    return 1
if ctx.cur_high > ctx.up * (1 + ctx.p0 / 100) and not ctx.high_acted:
    ctx.high_acted = True
    return -1
return 0
"""


# 注册 sweep 网格允许的参数名 (框架层 GRID_KEYS 只放引擎级, 策略参数自注册)
from ..core.sweep import register_grid_keys
register_grid_keys(set(DevTriggerStrategy.params_spec.keys()))
