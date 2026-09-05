from __future__ import annotations
"""strategies 子包: 策略目录(基类 + 注册表)

公开 API:
  StrategyBase             所有策略的基类
  get_strategy(name, **kw) 按 key 构造
  available_strategies()   所有可用 key
  register_strategy(name)  装饰器

当前已注册:
  - channel_deviation  通道偏离回撤 (与 evtrade.strategy.ChannelDeviationStrategy 等价)

用法 (未来 2b 阶段接 numba/CUDA 后):
  from evtrade.strategies import get_strategy
  s = get_strategy("channel_deviation", low1=1.5, low2=1.0, high1=1.5, high2=0.5)
  signal, info = s.check(cur, {"up": up, "dw": dw})
"""
# 触发装饰器副作用
from .base import (StrategyBase, register_strategy, get_strategy,
                   get_strategy_param_spec, available_strategies)
from . import channel_deviation  # noqa: F401  注册 channel_deviation
from . import example_breakout  # noqa: F401  注册 breakout (示例)

# DSL 转译器 (Python / numba / CUDA 三端)
from .dsl import compile_all, render_numba_body, render_cuda_body

__all__ = [
    "StrategyBase",
    "register_strategy", "get_strategy", "get_strategy_param_spec",
    "available_strategies",
    "compile_all", "render_numba_body", "render_cuda_body",
]
