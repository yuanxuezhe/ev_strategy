"""kernel_dsl 缓存键与失效行为 (三元 key + invalidate_dsl_cache)"""
from __future__ import annotations

import pytest

from evtrade.core import kernel_dsl as kd
from evtrade.core.kernel_dsl import (
    RENDERER_VERSION,
    _KERNEL_DSL_CACHE,
    build_dsl_kernel,
    dsl_kernel,
    invalidate_dsl_cache,
    _source_hash,
)
from evtrade.strategies.base import _STRATEGIES, StrategyBase, register_strategy


# ---------- 辅助: 注册一个临时的 DSL 策略 ----------


def _register_temp_dsl(name: str, body: str):
    """注册一个临时 DSL 策略; 返回 (name, cleanup_fn)"""
    spec = {"x": {"default": 0.0, "type": float}}

    @register_strategy(name)
    class _S(StrategyBase):
        params_spec = spec

        def compute_signal(self, ctx):
            """DSL body"""
            pass

    # 注入 docstring (DSL body)
    _S.compute_signal.__doc__ = body

    def _cleanup():
        _STRATEGIES.pop(name, None)
        # 顺手清掉缓存, 不影响其它测试
        invalidate_dsl_cache(name)

    return name, _cleanup


# ---------- 缓存键与失效 ----------


def test_source_hash_changes_with_docstring():
    """同一策略类, docstring 改一次, source hash 必须变化"""
    from evtrade.strategies import get_strategy_class
    cls = get_strategy_class("channel_deviation")
    h1 = _source_hash(cls)
    # 改 docstring, 再算一次
    orig_doc = cls.compute_signal.__doc__
    try:
        cls.compute_signal.__doc__ = orig_doc + "\n# tag\n"
        h2 = _source_hash(cls)
        assert h1 != h2
    finally:
        cls.compute_signal.__doc__ = orig_doc


def test_build_dsl_kernel_caches_by_source_hash():
    """同一 strategy_name + 同一 docstring 第二次返回同一 module 对象"""
    name, cleanup = _register_temp_dsl(
        "_cache_test_a", "return 0\n")
    try:
        invalidate_dsl_cache(name)
        m1 = build_dsl_kernel(name)
        m2 = build_dsl_kernel(name)
        assert m1 is m2
        # 缓存确实存了
        assert any(k[0] == name for k in _KERNEL_DSL_CACHE)
    finally:
        cleanup()


def test_invalidate_dsl_cache_clears_specific_strategy():
    name, cleanup = _register_temp_dsl("_cache_test_b", "return 0\n")
    try:
        invalidate_dsl_cache(name)
        m1 = build_dsl_kernel(name)
        n_before = len(_KERNEL_DSL_CACHE)
        assert n_before >= 1
        # 清除该策略的缓存
        removed = invalidate_dsl_cache(name)
        assert removed >= 1
        # 再次调用 -> 拿到新 module 对象
        m2 = build_dsl_kernel(name)
        assert m1 is not m2
    finally:
        cleanup()


def test_invalidate_dsl_cache_all():
    name1, c1 = _register_temp_dsl("_cache_test_c1", "return 0\n")
    name2, c2 = _register_temp_dsl("_cache_test_c2", "return 1\n")
    try:
        invalidate_dsl_cache()
        build_dsl_kernel(name1)
        build_dsl_kernel(name2)
        assert len(_KERNEL_DSL_CACHE) >= 2
        n = invalidate_dsl_cache()
        assert n >= 2
        assert _KERNEL_DSL_CACHE == {}
    finally:
        c1(); c2()


def test_dsl_kernel_channel_deviation_returns_frozen_kernel():
    """channel_deviation 特判保留: 返回冻结 kernel 本尊"""
    from evtrade.core import kernel as frozen_kernel
    assert dsl_kernel("channel_deviation") is frozen_kernel


def test_renderer_version_is_a_string():
    """RENDERER_VERSION 必须是字符串, 任何升级让其值变化"""
    assert isinstance(RENDERER_VERSION, str)
    assert RENDERER_VERSION  # 非空
