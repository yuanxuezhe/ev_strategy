from __future__ import annotations
"""array 后端选择器 (PyTorch 统一 CPU/GPU 路径)

get_xp(device) 返回 torch 后端:
  - "cpu"  -> torch.device("cpu")
  - "gpu"  -> torch.device("cuda")  (CUDA 不可用时 fallback cpu)
  - "cuda" -> torch.device("cuda")  (同上, 别名)
  - "auto" -> cuda 可用则 cuda, 否则 cpu

策略代码用 torch 写数组算子, 一份代码跑 CPU/GPU 两端。
PyTorch 在底层把 cumsum / where / unfold 等调用映射到 CPU SIMD 或 CUDA kernel。

bars 数组形状统一为 (B, T) —— B = batch (实盘/单线 =1, 网格扫描=N),
T = 时间序列长度。Engine.on_bars 维护 (1, T_lookback) 滑动窗口。
"""

import torch


def get_xp(device: str = "cpu") -> torch.device:
    """返回 torch 后端 device。

    - "cpu"  -> torch.device("cpu")
    - "gpu"  -> torch.device("cuda") if available else cpu (fallback + warning)
    - "cuda" -> torch.device("cuda") if available else cpu
    - "auto" -> cuda if available else cpu
    """
    if device in ("gpu", "cuda", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if device in ("gpu", "cuda"):
            import warnings
            warnings.warn(
                f"device={device!r} 但 CUDA 不可用; fallback to cpu",
                RuntimeWarning,
                stacklevel=2,
            )
        return torch.device("cpu")
    return torch.device("cpu")


def gpu_available() -> bool:
    """CUDA 是否可用 (用于测试 skip 判定, 不抛异常)"""
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(requested: str, gpu_ok: bool | None = None) -> str:
    """按 requested + CUDA 可用性解析设备 (三个子命令入口统一调用)。

    - "cpu": 直接返回 cpu
    - "gpu": CUDA 可用返回 gpu; 不可用抛 ValueError (带可操作提示)
    - "auto": 优先 gpu (需可用), 否则降级 cpu (调用方负责打 warning)
    """
    if gpu_ok is None:
        gpu_ok = gpu_available()
    if requested == "cpu":
        return "cpu"
    if requested == "gpu":
        if not gpu_ok:
            import torch
            raise ValueError(
                f"请求 --device gpu 但当前 torch 构建无 CUDA 支持 "
                f"(torch=={torch.__version__}, cuda={torch.version.cuda}); "
                f"改用 --device auto (自动降级) 或 --device cpu; "
                f"GPU 机器请先 `uv sync` 再 `bash scripts/sync-torch-cu.sh` "
                f"把 torch 覆盖到 cu128 wheel")
        return "gpu"
    return "gpu" if gpu_ok else "cpu"


def to_tensor(arr, device: torch.device | None = None) -> torch.Tensor:
    """numpy/list/tensor -> torch.Tensor (在指定 device 上)。

    device 为 None 时保持 arr 现有 device (tensor) 或 cpu (numpy/list)。
    """
    if isinstance(arr, torch.Tensor):
        return arr.to(device) if device is not None else arr
    return torch.as_tensor(arr, device=device)


def to_host(t: torch.Tensor) -> torch.Tensor:
    """tensor -> CPU tensor。"""
    return t.detach().cpu() if t.is_cuda else t.detach()
