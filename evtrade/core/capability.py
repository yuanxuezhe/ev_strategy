"""目标后端能力表 —— 调度层与 CLI 用 (单一事实源)

三端参数上限 + 字段上限不一致:
  - cpu (numba):  p0..p15  (KernelState.p0..p15)
  - gpu (cuda):   p0..p7   (_CUDA_SIG_FIELDS 写死)

策略作者写一个 10 参数 DSL 策略时, CPU sweep 跑通, 切 GPU 才失败;
应当由本模块在调度层做能力探测, 让 CLI/SDK 提前给出诊断或在
auto 模式下自动降级到 cpu。

注意: ctx 字段集合也由本模块给出 (与 strategies/dsl.py::_CUDA_SIG_FIELDS
数值上一致; 这里再列一份, 让调度层不必 import dsl 内部变量)。
"""
from __future__ import annotations

from typing import Literal

from ..strategies import get_strategy_param_spec

Device = Literal["cpu", "gpu"]


# 单一事实源: 三端参数上限
TARGET_CAPS: dict[str, dict] = {
    "cpu": {
        "max_params": 16,
        "label": "CPU/numba",
    },
    "gpu": {
        "max_params": 8,
        "label": "GPU/CUDA",
    },
}


def can_run(strategy_name: str, target: Device) -> tuple[bool, str]:
    """该策略是否能在 target 上编译/执行

    返回 (ok, reason): ok=False 时 reason 是给用户看的诊断 (>= 1 句);
    ok=True 时 reason 为空字符串。
    """
    cap = TARGET_CAPS[target]
    spec = get_strategy_param_spec(strategy_name)
    n = len(spec)
    if n > cap["max_params"]:
        keys = list(spec.keys())
        return False, (
            f"策略 {strategy_name!r} 有 {n} 个参数 "
            f"({', '.join(keys[:cap['max_params']])}, ...), "
            f"超过 {cap['label']} 上限 {cap['max_params']}; "
            f"请用其它 device 或拆分策略。"
        )
    return True, ""


def select_device(strategy_name: str,
                  requested: Device | Literal["auto"],
                  gpu_available: bool) -> Device:
    """按 requested + 能力探测选 device

    requested:
      - "gpu": 必须能在 gpu 上跑, 且环境有 cupy; 否则抛 ValueError
      - "cpu": 直接返回 cpu
      - "auto": 优先 gpu (需可用且策略兼容), 否则降级 cpu 并 log warning
    """
    if requested == "cpu":
        return "cpu"

    if requested == "gpu":
        ok, why = can_run(strategy_name, "gpu")
        if not ok:
            raise ValueError(f"无法在 gpu 上跑: {why}")
        if not gpu_available:
            raise ValueError("请求 gpu 但环境无可用 cupy/CUDA")
        return "gpu"

    # auto
    ok, _why = can_run(strategy_name, "gpu")
    if gpu_available and ok:
        return "gpu"
    return "cpu"


def gpu_available() -> bool:
    """环境探测: cupy 是否能 import 且有 CUDA 设备"""
    try:
        import cupy  # noqa: F401
        cp = cupy
        return bool(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return False
