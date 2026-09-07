"""调度层能力探测 (evtrade.core.capability)

覆盖:
  - can_run: 在 cpu/gpu 上的能力校验
  - select_device: auto / cpu / gpu 三种请求的最终 device
  - gpu_available: 环境探测
  - DSL 编译期参数校验: render_cuda_device_function 对 n_params > 8 即抛
"""
from __future__ import annotations

import pytest

from evtrade.core.capability import (
    TARGET_CAPS,
    can_run,
    gpu_available,
    select_device,
)
from evtrade.strategies.dsl import (
    CompileError,
    render_cuda_device_function,
)


def test_target_caps_limits():
    assert TARGET_CAPS["cpu"]["max_params"] == 16
    assert TARGET_CAPS["gpu"]["max_params"] == 8


def test_can_run_cpu_default_strategy():
    """channel_deviation 默认 4 参数, cpu / gpu 都兼容"""
    ok, why = can_run("channel_deviation", "cpu")
    assert ok and why == ""
    ok, why = can_run("channel_deviation", "gpu")
    assert ok and why == ""


def test_can_run_gpu_rejects_too_many_params(monkeypatch):
    """注册一个 10 参数的策略, gpu 应拒绝, cpu 应通过"""
    from evtrade.strategies.base import _STRATEGIES, StrategyBase, register_strategy

    @register_strategy("_cap_test_10p")
    class _S(StrategyBase):
        name = "_cap_test_10p"
        params_spec = {f"p{i}": {"default": 0.0, "type": float} for i in range(10)}
        state_spec = {}   # DSL 必填字段; capability 测试不依赖持久状态

    try:
        ok_cpu, _ = can_run("_cap_test_10p", "cpu")
        ok_gpu, why_gpu = can_run("_cap_test_10p", "gpu")
        assert ok_cpu is True
        assert ok_gpu is False
        assert "10 个参数" in why_gpu
        assert "_cap_test_10p" in why_gpu
    finally:
        _STRATEGIES.pop("_cap_test_10p", None)


def test_select_device_cpu_request():
    assert select_device("channel_deviation", "cpu", gpu_available()) == "cpu"


def test_select_device_auto_without_gpu_returns_cpu():
    """auto + 无 gpu 环境 -> cpu"""
    assert select_device("channel_deviation", "auto", gpu_available=False) == "cpu"


def test_select_device_gpu_request_without_gpu_raises():
    with pytest.raises(ValueError, match="gpu"):
        select_device("channel_deviation", "gpu", gpu_available=False)


def test_select_device_auto_with_gpu_for_compatible_strategy_returns_gpu():
    assert select_device("channel_deviation", "auto", gpu_available=True) == "gpu"


def test_select_device_auto_with_gpu_for_incompatible_strategy_returns_cpu():
    from evtrade.strategies.base import _STRATEGIES, StrategyBase, register_strategy

    @register_strategy("_cap_test_too_many")
    class _S(StrategyBase):
        name = "_cap_test_too_many"
        params_spec = {f"p{i}": {"default": 0.0, "type": float} for i in range(10)}
        state_spec = {}   # DSL 必填字段; capability 测试不依赖持久状态

    try:
        # 即使 gpu_available=True, 策略不兼容也应降级到 cpu
        assert select_device("_cap_test_too_many", "auto",
                             gpu_available=True) == "cpu"
    finally:
        _STRATEGIES.pop("_cap_test_too_many", None)


# ---------- render_cuda_device_function 编译期参数校验 ----------

def _make_cls_with_n_params(n: int):
    from evtrade.strategies.base import _STRATEGIES, StrategyBase, register_strategy

    name = f"_dsl_cuda_param_test_{n}"
    spec = {f"p{i}": {"default": 0.0, "type": float} for i in range(n)}

    @register_strategy(name)
    class _S(StrategyBase):
        pass

    _S.params_spec = spec
    _S.state_spec = {}   # DSL 必填字段; 测试用空 (无持久状态)

    def _cleanup():
        _STRATEGIES.pop(name, None)

    _S._cleanup = staticmethod(_cleanup)  # 让测试 fixture 用
    return name, _S, _cleanup


def test_render_cuda_device_function_rejects_9_params():
    name, cls, cleanup = _make_cls_with_n_params(9)
    try:
        # 必须给 compute_signal 一个合法的 docstring 才能让渲染走通
        def _sig(self, ctx):
            return 0
        _sig.__doc__ = "return 0\n"
        cls.compute_signal = _sig
        # 9 参数, 触发 CompileError 而不是到 GPU 运行时才发现
        with pytest.raises(CompileError, match="超过 CUDA 通用内核上限 8"):
            render_cuda_device_function(cls)
    finally:
        cleanup()


def test_render_cuda_device_function_accepts_8_params():
    name, cls, cleanup = _make_cls_with_n_params(8)
    try:
        # 8 参数应通过 (极限内)
        def _sig(self, ctx):
            return 0
        _sig.__doc__ = "return 0\n"
        cls.compute_signal = _sig
        out = render_cuda_device_function(cls)
        assert "p7" in out  # 签名里有 p7
        assert "p8" not in out  # 签名里没有 p8
    finally:
        cleanup()
