"""目标后端能力表 —— 调度层与 CLI 用 (单一事实源)"""
from __future__ import annotations

from typing import Literal

from ..backends import gpu_available as _gpu_available

Device = Literal["cpu", "gpu"]


TARGET_CAPS: dict[str, dict] = {
    "cpu": {"label": "CPU/numpy (xp=numpy)"},
    "gpu": {"label": "GPU/CuPy (xp=cupy)"},
}


def can_run(strategy_name: str, target: Device) -> tuple[bool, str]:
    """该策略是否能在 target 上执行 (唯一限制 = gpu_available)

    返回 (ok, reason): ok=False 时 reason 是给用户看的诊断 (>= 1 句);
    ok=True 时 reason 为空字符串。
    """
    return True, ""


def select_device(strategy_name: str,
                  requested: Device | Literal["auto"],
                  gpu_available: bool) -> Device:
    """按 requested + gpu_available 选 device

    requested:
      - "gpu": 必须有 cupy; 否则抛 ValueError
      - "cpu": 直接返回 cpu
      - "auto": 优先 gpu (需可用), 否则降级 cpu
    """
    if requested == "cpu":
        return "cpu"

    if requested == "gpu":
        if not gpu_available:
            raise ValueError("请求 gpu 但环境无可用 cupy/CUDA")
        return "gpu"

    if gpu_available:
        return "gpu"
    return "cpu"


def gpu_available() -> bool:
    """环境探测: cupy 是否能 import 且有 CUDA 设备 (委托 backends 单一真源)"""
    return _gpu_available()
