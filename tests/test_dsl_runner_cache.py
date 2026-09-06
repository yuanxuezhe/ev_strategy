"""make_python_runner 模块级缓存行为

覆盖:
  - 同一 (cls, method_name, source_hash) 第二次返回同 fn 对象
  - 跨实例: ChannelDeviationStrategy.__init__ 多份不重复 exec
  - invalidate_runner_cache: 全部 / 单类清空
  - source_hash 变化: 失效旧 runner
"""
from __future__ import annotations

import pytest

from evtrade.strategies import dsl as dsl_mod
from evtrade.strategies.dsl import (
    _RUNNER_BY_KEY,
    invalidate_runner_cache,
    make_python_runner,
)


def test_make_python_runner_caches_same_key():
    """同 cls + 同 docstring 第二次调用拿到同一 runner fn"""
    from evtrade.strategies import get_strategy_class
    invalidate_runner_cache()
    cls = get_strategy_class("channel_deviation")
    r1 = make_python_runner(cls)
    r2 = make_python_runner(cls)
    assert r1 is r2
    # 缓存里应当有一条
    assert len(_RUNNER_BY_KEY) >= 1


def test_make_python_runner_shared_across_instances():
    """ChannelDeviationStrategy 实例化两次, runner 应是同一个 fn 对象"""
    from evtrade.strategies import get_strategy

    invalidate_runner_cache()
    s1 = get_strategy("channel_deviation", low1=1.5)
    s2 = get_strategy("channel_deviation", low1=2.5)
    assert s1._runner is s2._runner


def test_invalidate_runner_cache_all():
    from evtrade.strategies import get_strategy_class
    invalidate_runner_cache()
    cls = get_strategy_class("channel_deviation")
    make_python_runner(cls)
    assert len(_RUNNER_BY_KEY) >= 1
    n = invalidate_runner_cache()
    assert n >= 1
    assert _RUNNER_BY_KEY == {}


def test_invalidate_runner_cache_specific_class():
    from evtrade.strategies.base import _STRATEGIES, StrategyBase, register_strategy

    invalidate_runner_cache()
    # 注册两个临时 DSL 策略
    @register_strategy("_runner_cache_a")
    class _A(StrategyBase):
        params_spec = {"x": {"default": 0.0, "type": float}}

        def compute_signal(self, ctx):
            """return 0"""
            pass

    @register_strategy("_runner_cache_b")
    class _B(StrategyBase):
        params_spec = {"x": {"default": 0.0, "type": float}}

        def compute_signal(self, ctx):
            """return 1"""
            pass

    try:
        make_python_runner(_A)
        make_python_runner(_B)
        assert len(_RUNNER_BY_KEY) >= 2
        n = invalidate_runner_cache(_A)
        assert n >= 1
        # _A 的 key 被清, _B 仍存
        keys = [k for k in _RUNNER_BY_KEY if k[0] is _A]
        assert keys == []
        keys_b = [k for k in _RUNNER_BY_KEY if k[0] is _B]
        assert keys_b, "_B 应当未被清"
    finally:
        _STRATEGIES.pop("_runner_cache_a", None)
        _STRATEGIES.pop("_runner_cache_b", None)
        invalidate_runner_cache()


def test_source_hash_change_invalidates_runner():
    """改 docstring -> 旧 runner 失效, 新 runner 编译"""
    from evtrade.strategies import get_strategy_class
    invalidate_runner_cache()
    cls = get_strategy_class("channel_deviation")
    orig_doc = cls.compute_signal.__doc__
    try:
        r1 = make_python_runner(cls)
        # 改 docstring -> source_hash 变 -> 缓存 miss -> 新 runner
        cls.compute_signal.__doc__ = orig_doc + "\n# tag\n"
        r2 = make_python_runner(cls)
        assert r1 is not r2
    finally:
        cls.compute_signal.__doc__ = orig_doc
        invalidate_runner_cache(cls)


def test_make_python_runner_validates_first():
    """make_python_runner 在 exec 前先 parse+validate,
    未知节点立即抛 CompileError, 不执行任意 docstring 代码。"""
    from evtrade.strategies.base import _STRATEGIES, StrategyBase, register_strategy
    from evtrade.strategies.dsl import CompileError

    @register_strategy("_runner_validate_test")
    class _S(StrategyBase):
        params_spec = {"x": {"default": 0.0, "type": float}}

        def compute_signal(self, ctx):
            """malicious attempt"""
            # ast.Subscript 节点不在白名单, 应当被 _validate 拒绝
            pass

    _S.compute_signal.__doc__ = "return ctx.cur_high[0]\n"
    try:
        with pytest.raises(CompileError):
            make_python_runner(_S)
    finally:
        _STRATEGIES.pop("_runner_validate_test", None)
        invalidate_runner_cache()
