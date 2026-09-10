from __future__ import annotations
"""GPU 环境探测 + 向量化桶预计算 (numpy 中间表示)

公开 API:
  - gpu_info:              GPU/CUDA 环境探测 (基于 torch.cuda)
  - precompute_ts_mark:    向量化桶时间戳 + 预热标记 (numpy 算术)
"""

import shutil
import subprocess
from collections import OrderedDict

import numpy as np

from .timeutils import resolve_period_seconds


def gpu_info() -> dict:
    """探测 GPU/CUDA 环境 (基于 torch.cuda; 任何缺失只记 None, 不抛异常)"""
    info = {"nvidia_gpu": None, "driver": None, "cuda_toolkit": None,
            "torch": None, "cuda_available": None}
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                                  "--format=csv,noheader"], capture_output=True,
                                 text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                name, _, drv = out.stdout.strip().partition(", ")
                info["nvidia_gpu"] = name or None
                info["driver"] = drv or None
        except Exception:
            pass
    info["cuda_toolkit"] = shutil.which("nvcc")
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                info["nvidia_gpu"] = info["nvidia_gpu"] or props.name
            except Exception:
                pass
    except ImportError:
        info["torch"] = None
    return info


def _encoded_to_epoch_np(t: np.ndarray) -> np.ndarray:
    y = t // 10_000_000_000
    mo = (t // 100_000_000) % 100
    d = (t // 1_000_000) % 100
    h = (t // 10_000) % 100
    mi = (t // 100) % 100
    s = t % 100
    yy = y - (mo <= 2)
    era = yy // 400
    yoe = yy - era * 400
    mp = (mo + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return (era * 146097 + doe - 719468) * 86400 + h * 3600 + mi * 60 + s


def _epoch_to_encoded_np(e: np.ndarray) -> np.ndarray:
    days = e // 86400
    sod = e % 86400
    h = sod // 3600
    mi = (sod % 3600) // 60
    s = sod % 60
    z = days + 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + np.where(mp < 10, 3, -9)
    y2 = y + (m <= 2)
    return ((((y2 * 100 + m) * 100 + d) * 100 + h) * 100 + mi) * 100 + s


_PRECOMPUTE_TS_MARK_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_PRECOMPUTE_TS_MARK_MAXSIZE = 32


def _precompute_cache_key(bars: dict, period: str, warmup_until: int):
    """缓存键: 用 id+bars 长度避免 dict 复用错命中"""
    return (id(bars), len(bars.get("stime", ())), period, int(warmup_until))


def precompute_ts_mark(bars: dict, period: str, warmup_until: int):
    """(周期, 预热阈值) -> (ts int64[n], mark int8[n]); 与策略参数无关, 每组共享

    桶算法与 timeutils.bucket_ts_encoded 同式 (本地锚定 epoch 取整, 任意 m/h/d 周期)。
    """
    key = _precompute_cache_key(bars, period, warmup_until)
    cached = _PRECOMPUTE_TS_MARK_CACHE.get(key)
    if cached is not None:
        _PRECOMPUTE_TS_MARK_CACHE.move_to_end(key)
        return cached
    stime = bars["stime"]
    P = resolve_period_seconds(period)
    e = _encoded_to_epoch_np(stime)
    e0 = (e // P) * P
    r = e - e0
    ts = np.where(r == 0,
                  _epoch_to_encoded_np(e0),
                  _epoch_to_encoded_np(e0 + P))
    mark = np.where(stime < warmup_until, 0, 1).astype(np.int8)
    out = (ts.astype(np.int64), mark)
    _PRECOMPUTE_TS_MARK_CACHE[key] = out
    while len(_PRECOMPUTE_TS_MARK_CACHE) > _PRECOMPUTE_TS_MARK_MAXSIZE:
        _PRECOMPUTE_TS_MARK_CACHE.popitem(last=False)
    return out


def invalidate_precompute_cache() -> int:
    """清除 precompute_ts_mark 缓存; 返回清除的条目数"""
    n = len(_PRECOMPUTE_TS_MARK_CACHE)
    _PRECOMPUTE_TS_MARK_CACHE.clear()
    return n
