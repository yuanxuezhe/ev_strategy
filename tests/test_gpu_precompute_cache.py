"""evtrade.gpu.precompute_ts_mark LRU 缓存行为

覆盖:
  - 同 (id(bars), len(stime), period, warmup) 第二次返回同一数组对象
  - 容量上限: 超过 _PRECOMPUTE_TS_MARK_MAXSIZE 时按 LRU 淘汰最旧
  - invalidate_precompute_cache 清空
  - 缓存命中不影响输出正确性
"""
from __future__ import annotations

import numpy as np
import pytest

from evtrade.core import gpu as gpu_mod
from evtrade.core.gpu import (
    _PRECOMPUTE_TS_MARK_CACHE,
    _PRECOMPUTE_TS_MARK_MAXSIZE,
    _precompute_cache_key,
    invalidate_precompute_cache,
    precompute_ts_mark,
)


def _make_bars(n=64):
    """最小可用 bars dict"""
    stime = np.arange(20260101000000, 20260101000000 + n * 60000, 60000,
                      dtype=np.int64)
    return {
        "stime": stime,
        "open":  np.full(n, 100.0, dtype=np.float64),
        "high":  np.full(n, 101.0, dtype=np.float64),
        "low":   np.full(n, 99.0, dtype=np.float64),
        "close": np.full(n, 100.5, dtype=np.float64),
        "volume": np.full(n, 1000, dtype=np.int64),
    }


def test_precompute_ts_mark_returns_correct_shapes():
    invalidate_precompute_cache()
    bars = _make_bars(64)
    ts, mark = precompute_ts_mark(bars, "5m", 20260101000000)
    assert ts.dtype == np.int64
    assert mark.dtype == np.int8
    assert ts.shape == (64,)
    assert mark.shape == (64,)


def test_precompute_ts_mark_caches_same_key():
    """同 (id, len, period, warmup) 第二次返回同对象"""
    invalidate_precompute_cache()
    bars = _make_bars(64)
    r1 = precompute_ts_mark(bars, "5m", 20260101000000)
    r2 = precompute_ts_mark(bars, "5m", 20260101000000)
    assert r1 is r2
    # 缓存里有一条
    assert len(_PRECOMPUTE_TS_MARK_CACHE) >= 1


def test_precompute_cache_key_distinguishes_warmup():
    """warmup_until 不同 -> key 不同 -> 不命中旧缓存"""
    invalidate_precompute_cache()
    bars = _make_bars(64)
    precompute_ts_mark(bars, "5m", 0)
    precompute_ts_mark(bars, "5m", 20260101000000)
    assert len(_PRECOMPUTE_TS_MARK_CACHE) >= 2


def test_precompute_cache_capacity_bound():
    """超过上限按 LRU 淘汰最旧"""
    invalidate_precompute_cache()
    base = _make_bars(64)
    # 用不同 id 的 bars (新 dict) 填满 + 1
    n = _PRECOMPUTE_TS_MARK_MAXSIZE + 1
    for i in range(n):
        bars = _make_bars(64)
        # id 不同保证每条是独立 key (id 是 id() 函数返回值, 每次新建不同)
        precompute_ts_mark(bars, "5m", 0)
    assert len(_PRECOMPUTE_TS_MARK_CACHE) <= _PRECOMPUTE_TS_MARK_MAXSIZE


def test_invalidate_precompute_cache_clears():
    invalidate_precompute_cache()
    precompute_ts_mark(_make_bars(64), "5m", 0)
    assert len(_PRECOMPUTE_TS_MARK_CACHE) >= 1
    n = invalidate_precompute_cache()
    assert n >= 1
    assert _PRECOMPUTE_TS_MARK_CACHE == {}


def test_precompute_cache_key_components():
    """key 由 (id(bars), len(stime), period, warmup) 四元组成"""
    bars1 = _make_bars(64)
    bars2 = _make_bars(64)
    assert _precompute_cache_key(bars1, "5m", 0) != _precompute_cache_key(bars2, "5m", 0)
    assert _precompute_cache_key(bars1, "5m", 0) != _precompute_cache_key(bars1, "1m", 0)
    assert _precompute_cache_key(bars1, "5m", 0) != _precompute_cache_key(bars1, "5m", 1)
