"""PyTorch 后端基础 (pytorch-unified-strategy, 2026-09-10)

锁定:
  - evtrade.backends.get_xp(device) -> torch.device
  - evtrade.backends.gpu_available -> torch.cuda.is_available
  - cupy 已下线 (import cupy 抛 ImportError)
  - 旧 device="gpu" 字符串仍接受 (转 torch.device("cuda"))
  - to_tensor / to_host 工具函数
"""
from __future__ import annotations

import importlib

import pytest
import torch

import evtrade.backends as backends


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
    # 项目代码不应有 "import cupy" 出现在 evtrade/ 任何 .py
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


def test_xp_ema_torch_basic():
    """xp_ema_torch 接受 (B, T) tensor 输入"""
    from evtrade.indicators.ema import xp_ema_torch
    v = torch.randn(1, 50, dtype=torch.float64)
    out = xp_ema_torch(v, 21)
    assert out.shape == (1, 50)
    # 前 20 个为 NaN (period=21, p-1=20)
    assert torch.isnan(out[0, :20]).all()
    # 第 20 个 (idx) 是 SMA seed, 不为 NaN
    assert not torch.isnan(out[0, 20])


def test_xp_ema_torch_1d_input_unsqueeze():
    """1D (T,) 输入自动 unsqueeze 为 (1, T)"""
    from evtrade.indicators.ema import xp_ema_torch
    v = torch.randn(50, dtype=torch.float64)
    out = xp_ema_torch(v, 21)
    assert out.shape == (50,)


def test_xp_ema_torch_per_row_period():
    """per-row period: (B,) 形状 period"""
    from evtrade.indicators.ema import xp_ema_torch
    v = torch.randn(3, 100, dtype=torch.float64)
    periods = torch.tensor([5, 21, 50])
    out = xp_ema_torch(v, periods)
    assert out.shape == (3, 100)
    # period=5 行: 前 4 个 NaN
    assert torch.isnan(out[0, :4]).all()
    assert not torch.isnan(out[0, 4])
    # period=50 行: 前 49 个 NaN
    assert torch.isnan(out[2, :49]).all()
