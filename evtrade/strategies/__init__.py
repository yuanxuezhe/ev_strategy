from __future__ import annotations
"""strategies 子包: 策略目录 (统一 CPU/GPU 契约; DSL 已下线)

公开 API:
  VectorizedStrategy         所有策略的基类 (CPU/GPU 同一份契约)
  get_strategy(name, **kw)   按 key 构造
  get_strategy_class(name)    按 key 拿类 (不实例化)
  get_strategy_param_spec(name) 取 params_spec (sweep grid / CLI 校验)
  available_strategies()     所有可用 key
  register_strategy(name)    装饰器

当前已注册:
  - channel_deviation    通道偏离回撤 (混合向量化 + Python FSM)
  - ma_crossover         双均线交叉 (纯向量化, CuPy 路径)

用法 (DSL 三端同源已下线, 唯一策略契约 compute_signals(xp, bars, params)):
  from evtrade.strategies import get_strategy
  s = get_strategy("channel_deviation", low1=1.5, tf1=21)
  sig = s.compute_signals(xp, bars, s.params)
"""
# 触发装饰器副作用 (注册到 vectorized_base._STRATEGIES)
from .vectorized_base import (
    VectorizedStrategy, register_strategy, get_strategy,
    get_strategy_class, get_strategy_param_spec, available_strategies,
)
from . import channel_deviation  # noqa: F401  注册 channel_deviation
from . import ma_crossover       # noqa: F401  注册 ma_crossover

__all__ = [
    "VectorizedStrategy",
    "register_strategy", "get_strategy", "get_strategy_class",
    "get_strategy_param_spec", "available_strategies",
]
