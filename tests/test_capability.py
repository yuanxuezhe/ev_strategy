"""调度层能力探测 (evtrade.core.capability)

DSL/numba/CUDA 已下线. CPU 和 GPU 走同一条 vectorized 路径,
能力探测只剩两件事:
  - can_run: DSL 删除后策略永远能跑 (返回 True, "")
  - select_device: 按 requested + gpu_available 决定 cpu/gpu
  - gpu_available: 环境探测 (cupy 是否能 import)

策略参数上限不再按设备分化 (DSL 三端投影不存在了)。
"""
from __future__ import annotations

import pytest

from evtrade.core.capability import (
    TARGET_CAPS,
    can_run,
    gpu_available,
    select_device,
)


def test_target_caps_lists_cpu_gpu():
    """DSL 删除后 TARGET_CAPS 只剩 cpu / gpu 两条 entry, 没有 max_params 限制"""
    assert set(TARGET_CAPS.keys()) == {"cpu", "gpu"}
    assert "label" in TARGET_CAPS["cpu"]
    assert "label" in TARGET_CAPS["gpu"]
    # DSL 三端投影下线: 不再有 max_params / max_state 字段
    assert "max_params" not in TARGET_CAPS["cpu"]
    assert "max_params" not in TARGET_CAPS["gpu"]


def test_can_run_always_true_for_known_strategy():
    """DSL 删除后, 所有策略在 cpu/gpu 上都兼容"""
    for dev in ("cpu", "gpu"):
        ok, why = can_run("channel_deviation", dev)
        assert ok is True
        assert why == ""


def test_can_run_true_for_arbitrary_name():
    """DSL 删除后, 即便是未知策略名, can_run 也直接通过
    (注册校验在 get_strategy 时已经发生, 这里只是设备能力)"""
    ok, why = can_run("does_not_exist_strategy", "gpu")
    assert ok is True
    assert why == ""


def test_select_device_cpu_request_returns_cpu():
    assert select_device("channel_deviation", "cpu", gpu_available()) == "cpu"


def test_select_device_auto_without_gpu_returns_cpu():
    """auto + 无 gpu 环境 -> cpu"""
    assert select_device("channel_deviation", "auto", gpu_available=False) == "cpu"


def test_select_device_gpu_request_without_gpu_raises():
    with pytest.raises(ValueError, match="gpu"):
        select_device("channel_deviation", "gpu", gpu_available=False)


def test_select_device_auto_with_gpu_returns_gpu():
    """auto + 有 gpu -> gpu (不管策略名是什么, DSL 删除后无兼容性差异)"""
    assert select_device("channel_deviation", "auto", gpu_available=True) == "gpu"
    assert select_device("any_other_strategy", "auto", gpu_available=True) == "gpu"


def test_gpu_available_returns_bool():
    """环境探测返回 bool (即使 cupy 不可用也不抛错)"""
    result = gpu_available()
    assert isinstance(result, bool)