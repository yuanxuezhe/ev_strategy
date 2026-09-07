from __future__ import annotations
"""strategies 子包: 策略目录(基类 + 注册表)

公开 API:
  StrategyBase             所有策略的基类
  get_strategy(name, **kw) 按 key 构造
  get_strategy_class(name)  按 key 拿类 (不实例化, 编译期探针用)
  available_strategies()   所有可用 key
  register_strategy(name)  装饰器

当前已注册:
  - channel_deviation  通道偏离回撤 (与 evtrade.strategy.ChannelDeviationStrategy 等价)

用法 (DSL 三端同源: Python / numba / CUDA):
  from evtrade.strategies import get_strategy
  s = get_strategy("strategy_name", k1=v1, k2=v2)   # 具体键名见策略 params_spec
  signal, info = s.check(cur, {"up": up, "dw": dw})  # 或 positional up, dw
"""
# 触发装饰器副作用
from .base import (StrategyBase, register_strategy, get_strategy,
                   get_strategy_class, get_strategy_param_spec,
                   get_strategy_state_spec,
                   available_strategies)
from . import channel_deviation  # noqa: F401  注册 channel_deviation
from . import example_breakout  # noqa: F401  注册 breakout (示例, 仅参考引擎)
from . import example_dev_trigger  # noqa: F401  注册 dev_trigger (DSL 三端同源示例)

# DSL 转译器 (Python / numba / CUDA 三端)
from .dsl import (compile_all, render_numba_body, render_cuda_body,
                  render_numba_state_body, render_cuda_device_function,
                  build_ctx_to_kernel_map, build_cuda_sig_fields,
                  build_cuda_device_header, make_dsl_ctx,
                  DSLCtx, dsl_check)

__all__ = [
    "StrategyBase",
    "register_strategy", "get_strategy", "get_strategy_class",
    "get_strategy_param_spec", "get_strategy_state_spec",
    "available_strategies",
    "compile_all", "render_numba_body", "render_cuda_body",
    "render_numba_state_body", "render_cuda_device_function",
    "build_ctx_to_kernel_map", "build_cuda_sig_fields",
    "build_cuda_device_header", "make_dsl_ctx",
    "DSLCtx", "dsl_check",
]
