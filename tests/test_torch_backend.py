"""PyTorch 后端 + 设备解析 (pytorch-unified-strategy + add-gpu-extra-pyproject)

锁定:
  - evtrade.backends.get_xp(device) -> torch.device
  - evtrade.backends.gpu_available -> torch.cuda.is_available
  - evtrade.backends.resolve_device(requested, gpu_ok) -> str (cpu/gpu)
  - cupy 已下线 (import cupy 不应出现在 evtrade/ 代码)
  - 旧 device="gpu" 字符串仍接受 (转 torch.device("cuda"))
  - to_tensor / to_host 工具函数
  - --device gpu ValueError 文案含 GPU 安装引导 (uv sync + bash scripts/sync-torch-cu.sh)
"""
from __future__ import annotations

import importlib

import pytest
import torch

import evtrade.backends as backends
from evtrade.backends import gpu_available, resolve_device


# ============ get_xp / torch.device ============

def test_get_xp_returns_torch_device():
    """get_xp 必须返回 torch.device (不是 numpy/cupy 模块)"""
    xp_cpu = backends.get_xp("cpu")
    assert isinstance(xp_cpu, torch.device)
    assert xp_cpu.type == "cpu"

    # gpu/cuda/auto 都返回 device (可能 fallback cpu)
    xp_auto = backends.get_xp("auto")
    assert isinstance(xp_auto, torch.device)


def test_get_xp_gpu_fallback_to_cpu():
    """CUDA 不可用时, device='gpu'/'cuda' fallback cpu + warning"""
    if torch.cuda.is_available():
        pytest.skip("CUDA 可用; 此测试针对 fallback 路径")
    with pytest.warns(RuntimeWarning, match="fallback to cpu"):
        xp = backends.get_xp("gpu")
    assert xp.type == "cpu"


def test_gpu_available_calls_torch_cuda():
    """gpu_available 直接走 torch.cuda.is_available"""
    expected = torch.cuda.is_available()
    assert backends.gpu_available() is expected


def test_cupy_import_fails_or_unused():
    """cupy 已下线; 项目不应 import cupy

    注: cupy 可能作为可选依赖存在于环境, 但项目代码不应 import。
    """
    import subprocess
    r = subprocess.run(
        ["grep", "-r", "import cupy", "evtrade/"],
        capture_output=True, text=True
    )
    # 应返回 1 (无匹配)
    assert r.returncode != 0, f"项目代码仍引用 cupy: {r.stdout}"


def test_to_tensor_from_numpy():
    """numpy 数组 -> torch.Tensor (默认 cpu)"""
    import numpy as np
    arr = np.array([1.0, 2.0, 3.0])
    t = backends.to_tensor(arr)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float64  # numpy float64 默认
    assert t.tolist() == [1.0, 2.0, 3.0]


def test_to_tensor_existing_tensor_no_copy():
    """已是 tensor 时, to_tensor 不复制 (除非指定 device)"""
    t0 = torch.tensor([1.0, 2.0])
    t1 = backends.to_tensor(t0)
    assert t1 is t0
    # 指定 device 时仍走 to()
    t2 = backends.to_tensor(t0, device=torch.device("cpu"))
    assert t2 is not t0 or t0.device == torch.device("cpu")


def test_to_host_returns_cpu_tensor():
    """to_host 把任意 device tensor 拉回 cpu"""
    t = torch.tensor([1.0, 2.0])  # 默认 cpu
    h = backends.to_host(t)
    assert isinstance(h, torch.Tensor)
    assert h.device.type == "cpu"


# ============ resolve_device (ex test_device_resolution.py) ============

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


def test_resolve_device_gpu_error_message_includes_install_hints():
    """2026-09-13 drop-gpu-extra-cpu-default: --device gpu 报错的文案必须含 GPU 安装引导

    当 torch 是 CPU wheel 时, ValueError 文案需提示:
      - 当前 torch 构建无 CUDA 支持 (含 torch.__version__ / cuda 字段)
      - 改用 --device auto 或 --device cpu
      - GPU 机器请 `uv sync` 后 `bash scripts/sync-torch-cu.sh` 覆盖到 cu128
        (不再提示 `--extra gpu`, 该 extra 已下线)
    """
    with pytest.raises(ValueError) as ei:
        resolve_device("gpu", gpu_ok=False)
    msg = str(ei.value)
    assert "sync-torch-cu.sh" in msg, (
        f"ValueError 文案必须含 GPU 安装引导 (sync-torch-cu.sh), 实际: {msg!r}")
    assert "--extra gpu" not in msg, (
        "gpu extra 已下线, 文案不应再提示 --extra gpu")
    assert "auto" in msg and "cpu" in msg, "文案需提示改用 --device auto/cpu"


def test_resolve_device_gpu_request_with_gpu_returns_gpu():
    assert resolve_device("gpu", gpu_ok=True) == "gpu"
