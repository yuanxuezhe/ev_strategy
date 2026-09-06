"""evtrade.gpu GPU 缓存键 + invalidate_gpu_cache 行为"""
from __future__ import annotations

import pytest

from evtrade.core import gpu as gpu_mod
from evtrade.core.gpu import (
    GPU_RENDERER_VERSION,
    _GENERIC_KERNEL_CACHE,
    _source_hash_gpu,
    invalidate_gpu_cache,
)
from evtrade.strategies.base import _STRATEGIES, StrategyBase, register_strategy


# ---------- 辅助 ----------


def _register_temp_dsl(name: str, body: str):
    @register_strategy(name)
    class _S(StrategyBase):
        params_spec = {"x": {"default": 0.0, "type": float}}

        def compute_signal(self, ctx):
            """DSL body"""
            pass

    _S.compute_signal.__doc__ = body

    def _cleanup():
        _STRATEGIES.pop(name, None)

    return name, _cleanup


# ---------- 基础常量与函数 ----------


def test_renderer_version_is_string():
    assert isinstance(GPU_RENDERER_VERSION, str)
    assert GPU_RENDERER_VERSION


def test_source_hash_changes_with_docstring():
    """docstring 改一次, hash 必须变"""
    from evtrade.strategies import get_strategy_class
    cls = get_strategy_class("channel_deviation")
    h1 = _source_hash_gpu(cls)
    orig_doc = cls.compute_signal.__doc__
    try:
        cls.compute_signal.__doc__ = orig_doc + "\n# tag\n"
        h2 = _source_hash_gpu(cls)
        assert h1 != h2
    finally:
        cls.compute_signal.__doc__ = orig_doc


def test_invalidate_gpu_cache_specific_strategy():
    """invalidate_gpu_cache(name) 只清特定策略"""
    name, cleanup = _register_temp_dsl("_gpu_cache_a", "return 0\n")
    try:
        invalidate_gpu_cache()
        # 手动塞个条目 (避免真调 _compile_generic_kernel 需要 cupy)
        key = (name, "fake_hash", "70", GPU_RENDERER_VERSION)
        _GENERIC_KERNEL_CACHE[key] = object()
        assert len(_GENERIC_KERNEL_CACHE) >= 1
        removed = invalidate_gpu_cache(name)
        assert removed == 1
        # 不影响其它 key
        other_key = ("other", "h", "70", "v0")
        _GENERIC_KERNEL_CACHE[other_key] = object()
        invalidate_gpu_cache(name)
        assert other_key in _GENERIC_KERNEL_CACHE
    finally:
        invalidate_gpu_cache()
        cleanup()


def test_invalidate_gpu_cache_all():
    """invalidate_gpu_cache() 清空全部"""
    _GENERIC_KERNEL_CACHE[("a", "h", "70", "v0")] = object()
    _GENERIC_KERNEL_CACHE[("b", "h", "80", "v0")] = object()
    n = invalidate_gpu_cache()
    assert n >= 2
    assert _GENERIC_KERNEL_CACHE == {}


def test_gpu_module_no_longer_uses_legacy_cache_name():
    """重构后不应再使用旧名字 _generic_kernel_cache (避免代码混淆)"""
    assert not hasattr(gpu_mod, "_generic_kernel_cache"), (
        "应已迁移到 _GENERIC_KERNEL_CACHE")
    assert hasattr(gpu_mod, "_GENERIC_KERNEL_CACHE")
