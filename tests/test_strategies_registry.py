"""策略注册表 + ListBarFeed (合并自 test_harness + test_strategy_class_access)

锁定:
  - get_strategy_class: 返类不触发 __init__ (避免 params_spec 校验副作用)
  - get_strategy: 仍按合同实例化 (dict / kwargs / 混合)
  - ListBarFeed passthrough yield Bar 列表元素
"""
from __future__ import annotations

import pytest

from evtrade.strategies import get_strategy, get_strategy_class
from evtrade.strategies.vectorized_base import _STRATEGIES


# ============ get_strategy_class ============

def test_get_strategy_class_returns_registered_class():
    cls = get_strategy_class("channel_deviation")
    assert cls is _STRATEGIES["channel_deviation"]


def test_get_strategy_class_unknown_raises():
    with pytest.raises(ValueError, match="未知策略"):
        get_strategy_class("does_not_exist")


def test_get_strategy_class_does_not_instantiate():
    """get_strategy_class 不应触发 __init__ (VectorizedStrategy 实例化会跑 params_spec 校验)"""
    import evtrade.strategies.channel_deviation as cd

    init_calls = []

    orig_init = cd.ChannelDeviationStrategy.__init__

    def spy(self, *a, **kw):
        init_calls.append(self)
        return orig_init(self, *a, **kw)

    cd.ChannelDeviationStrategy.__init__ = spy
    try:
        cls = get_strategy_class("channel_deviation")
        assert init_calls == [], f"get_strategy_class 实例化了: {len(init_calls)} 次"
        assert cls is cd.ChannelDeviationStrategy
    finally:
        cd.ChannelDeviationStrategy.__init__ = orig_init


def test_get_strategy_still_instantiates():
    """对照: get_strategy 仍按合同实例化 (带或不带 params 都行)"""
    s = get_strategy("channel_deviation", low1=2.0)
    assert s.low1 == 2.0


# ============ ListBarFeed ============

def test_list_bar_feed_passthrough():
    """ListBarFeed.stream() 按序 yield 列表元素"""
    from evtrade.core._harness import ListBarFeed
    fake = [object() for _ in range(5)]
    out = list(ListBarFeed(fake).stream())
    assert out == fake


def test_sweep_module_imports():
    """sweep 模块可导入 + 含 run_one_vectorized (兼容测试基础设施)"""
    import evtrade.core.sweep as sweep_mod
    assert hasattr(sweep_mod, "run_one_vectorized")
