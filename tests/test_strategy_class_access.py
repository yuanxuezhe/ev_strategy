"""strategies.base.get_strategy_class 行为"""
from __future__ import annotations

import pytest

from evtrade.strategies import get_strategy, get_strategy_class
from evtrade.strategies.base import _STRATEGIES


def test_get_strategy_class_returns_registered_class():
    cls = get_strategy_class("channel_deviation")
    assert cls is _STRATEGIES["channel_deviation"]


def test_get_strategy_class_unknown_raises():
    with pytest.raises(ValueError, match="未知策略"):
        get_strategy_class("does_not_exist")


def test_get_strategy_class_does_not_instantiate():
    """get_strategy_class 不应触发 __init__ (通道偏离会执行 DSL 编译)"""
    import evtrade.strategies.channel_deviation as cd

    # 在 _runner 真正首次调用前, ChannelDeviationStrategy 没有 ._runner 字段
    # 实例化会让 __init__ 把 _runner 挂上 -> 用 spy 检测
    init_calls = []

    orig_init = cd.ChannelDeviationStrategy.__init__

    def spy(self, *a, **kw):
        init_calls.append(self)
        return orig_init(self, *a, **kw)

    cd.ChannelDeviationStrategy.__init__ = spy
    try:
        cls = get_strategy_class("channel_deviation")
        assert init_calls == [], f"get_strategy_class 实例化了: {len(init_calls)} 次"
        # 但仍然返回的是类本身
        assert cls is cd.ChannelDeviationStrategy
    finally:
        cd.ChannelDeviationStrategy.__init__ = orig_init


def test_get_strategy_still_instantiates():
    """对照: get_strategy 仍按合同实例化 (带或不带 params 都行)"""
    s = get_strategy("channel_deviation", low1=2.0)
    assert s.low1 == 2.0
