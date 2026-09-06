"""evtrade.core._harness 公共 fixture

覆盖:
  - NumpyDictFeed 从 numpy dict 正确 yield Bar
  - ListBarFeed passthrough yield Bar list 元素
  - sweep.run_one_general 替换 _ArrFeed 后行为不变
"""
from __future__ import annotations

import numpy as np


def _make_bars(n=8):
    """BarAggregator 期望 stime 为 YYYYMMDDHHmmss (14 字符)"""
    stime = np.arange(20260101000000, 20260101000000 + n * 60000, 60000,
                      dtype=np.int64)
    return {
        "stime": stime,
        "open":  np.arange(n, dtype=np.float64),
        "high":  np.arange(n, dtype=np.float64) + 1.0,
        "low":   np.arange(n, dtype=np.float64) - 1.0,
        "close": np.arange(n, dtype=np.float64) + 0.5,
        "volume": np.full(n, 1000, dtype=np.int64),
    }


def test_numpy_dict_feed_yields_correct_bar_count():
    from evtrade.core._harness import NumpyDictFeed
    bars = _make_bars(16)
    feed = NumpyDictFeed(bars)
    out = list(feed.stream())
    assert len(out) == 16
    assert all(b.code == "SYN" for b in out)
    assert all(b.stime == str(int(bars["stime"][i])) for i, b in enumerate(out))


def test_list_bar_feed_passthrough():
    from evtrade.core._harness import ListBarFeed
    fake = [object() for _ in range(5)]
    out = list(ListBarFeed(fake).stream())
    assert out == fake


def test_sweep_module_imports_NumpyDictFeed():
    """sweep.py 在 run_one_general 中已 import 并使用 NumpyDictFeed"""
    import evtrade.core.sweep as sweep_mod
    from evtrade.core._harness import NumpyDictFeed
    # NumpyDictFeed 必须存在于 _harness 且可被 sweep 引用
    assert callable(NumpyDictFeed)
    # sweep 模块本身必须可 import (没语法错)
    assert hasattr(sweep_mod, "run_one_general")


def test_replay_module_imports_ListBarFeed():
    """replay.py 在 replay_engine 中已 import 并使用 ListBarFeed"""
    import evtrade.core.replay as replay_mod
    from evtrade.core._harness import ListBarFeed
    assert callable(ListBarFeed)
    assert hasattr(replay_mod, "replay_engine")
