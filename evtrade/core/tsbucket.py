from __future__ import annotations
"""桶级 ts/mark 预计算 (纯 numpy, 设备无关)

公开 API:
  - precompute_ts_mark: 向量化桶时间戳 + 预热标记 (numpy 算术, LRU 缓存)
  - invalidate_precompute_cache: 清空缓存

历法运算 (encoded_to_epoch_np / epoch_to_encoded_np) 单一真源在 core/timeutils.py。
"""

from collections import OrderedDict

import numpy as np

from .timeutils import (
    encoded_to_epoch_np, epoch_to_encoded_np, resolve_period_seconds,
)

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
    e = encoded_to_epoch_np(stime)
    e0 = (e // P) * P
    r = e - e0
    ts = np.where(r == 0,
                  epoch_to_encoded_np(e0),
                  epoch_to_encoded_np(e0 + P))
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
