"""设备解析 (evtrade.backends.resolve_device / gpu_available)

core/capability.py 已删除, 能力探测收编到 backends:
  - resolve_device: 按 requested + gpu_available 决定 cpu/gpu
      cpu  -> cpu
      gpu  -> gpu (无 CUDA 抛 ValueError)
      auto -> gpu 可用则 gpu, 否则 cpu
  - gpu_available: 环境探测 (torch CUDA 是否可用, 返回 bool)
"""
from __future__ import annotations

import pytest

from evtrade.backends import gpu_available, resolve_device


def test_resolve_device_cpu_request_returns_cpu():
    assert resolve_device("cpu", gpu_ok=True) == "cpu"
    assert resolve_device("cpu", gpu_ok=False) == "cpu"


def test_resolve_device_auto_without_gpu_returns_cpu():
    """auto + 无 gpu 环境 -> cpu"""
    assert resolve_device("auto", gpu_ok=False) == "cpu"


def test_resolve_device_auto_with_gpu_returns_gpu():
    """auto + 有 gpu -> gpu"""
    assert resolve_device("auto", gpu_ok=True) == "gpu"


def test_resolve_device_gpu_request_without_gpu_raises():
    with pytest.raises(ValueError, match="gpu"):
        resolve_device("gpu", gpu_ok=False)


def test_resolve_device_gpu_request_with_gpu_returns_gpu():
    assert resolve_device("gpu", gpu_ok=True) == "gpu"


def test_gpu_available_returns_bool():
    """环境探测返回 bool (即使 torch CUDA 不可用也不抛错)"""
    result = gpu_available()
    assert isinstance(result, bool)
