from __future__ import annotations
"""strategies 子包: 策略目录 (strategy-step-only, 2026-09-10)

公开 API:
  VectorizedStrategy         所有策略的基类
  get_strategy(name, **kw)   按 key 构造
  get_strategy_class(name)    按 key 拿类 (不实例化)
  get_strategy_param_spec(name) 取 params_spec (sweep grid / CLI 校验)
  available_strategies()     所有可用 key
  register_strategy(name)    装饰器

当前已注册:
  - channel_deviation    通道偏离回撤 (stateful step + EMA 通道 + FSM)
  - ma_crossover         双均线交叉 (stateful step)

用法 (strategy-step-only, 唯一策略契约 step(state, bar, params) -> (state, sig)):
  from evtrade.strategies import get_strategy
  s = get_strategy("channel_deviation", low1=1.5, tf1=21)
  state = s.init_state(s.params)
  for bar in bars:
      state, sig = s.step(state, bar, s.params)
  # engine (VectorizedEngine / Engine.on_bars) 内部循环调用
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
