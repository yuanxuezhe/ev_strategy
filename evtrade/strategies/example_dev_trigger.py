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
"""
from .base import StrategyBase, register_strategy
from .dsl import make_dsl_ctx, dsl_check


@register_strategy("dev_trigger")
class DevTriggerStrategy(StrategyBase):
    """轨道触发策略 (有状态, DSL 版)

    params (dict 或 kwargs):
      entry_dev: 触发深度 (%): 低点 < 下轨×(1-p0/100) 买 / 高点 > 上轨×(1+p0/100) 卖
    """

    params_spec = {
        "entry_dev": {"default": 0.8, "type": float, "min": 0.0, "max": 50.0},
    }

    def __init__(self, params: dict | None = None, **kwargs):
        super().__init__(params=params, **kwargs)
        self._ctx = make_dsl_ctx(type(self))   # 状态字段跨桶持久

    def compute_signal(self, ctx):
        """DSL 主逻辑 (dev_trigger)"""
        pass  # DSL 注入

    def check(self, cur, up_or_indicators, dw=None):
        """兼容 (cur, up, dw) 与 (cur, dict) 两种调用; 引擎层零逻辑"""
        if isinstance(up_or_indicators, dict):
            up = up_or_indicators.get("up")
            dw_val = up_or_indicators.get("dw")
        else:
            up = up_or_indicators
            dw_val = dw
        if up is None or dw_val is None or up == 0 or dw_val == 0:
            return None, {}
        sig = dsl_check(self, self._ctx, cur, up, dw_val)
        return {1: "BUY", -1: "SELL"}.get(sig), {}


DevTriggerStrategy.compute_signal.__doc__ = """\
if ctx.up != ctx.up or ctx.dw != ctx.dw or ctx.up == 0 or ctx.dw == 0:
    return 0
if ctx.cur_ts != ctx._bucket_ts:
    ctx._bucket_ts = ctx.cur_ts
    ctx._low_acted = False
    ctx._high_acted = False
if ctx.cur_low < ctx.dw * (1 - ctx.p0 / 100) and not ctx._low_acted:
    ctx._low_acted = True
    return 1
if ctx.cur_high > ctx.up * (1 + ctx.p0 / 100) and not ctx._high_acted:
    ctx._high_acted = True
    return -1
return 0
"""


# 注册 sweep 网格允许的参数名 (框架层 GRID_KEYS 只放引擎级, 策略参数自注册)
from ..core.sweep import register_grid_keys
register_grid_keys(set(DevTriggerStrategy.params_spec.keys()))
