from __future__ import annotations
"""array 后端选择器 (CuPy 统一 CPU/GPU 路径)

vectorized 引擎通过 `get_xp(device)` 拿到 array 后端:
  - device="cpu"  -> numpy (np)
  - device="gpu"  -> cupy  (cp)

策略代码用 `xp` 写数组算子, 一份代码跑两端。CuPy 在底层把 cp.cumsum /
cp.where 等调用映射到预编译的 CUDA kernel, 无需手写 C++ / DSL 渲染。

适用: 可向量化的策略 (MA 交叉 / 突破 / 协方差等标准数组运算)。
不适用: 逐 bar 状态机 (channel_deviation 的 if/elif 锁存) —— 策略在
compute_signals 内通过 .get() 拉回 host 维护 Python FSM, 见 channel_deviation.py。
"""


def get_xp(device: str = "cpu"):
    """返回 array 后端模块 (numpy 或 cupy)

    device="gpu" 时复用 core.gpu._ensure_cupy 探测 cupy (pip 的 nvidia-*-cu12
    轮子提供 DLL, 见 gpu.py); 探测失败抛 ImportError。
    """
    if device == "gpu":
        from .core.gpu import _ensure_cupy
        return _ensure_cupy()
    import numpy as np
    return np


def gpu_available() -> bool:
    """cupy 是否可用 (用于测试 skip 判定, 不抛异常)"""
    try:
        from .core.gpu import _ensure_cupy
        _ensure_cupy()
        return True
    except Exception:
        return False
