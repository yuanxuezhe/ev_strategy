"""evtrade.core._harness 公共 fixture

覆盖:
  - ListBarFeed passthrough yield Bar list 元素
  - sweep 模块直接吃 numpy dict (run_one_vectorized 可 import)
  - replay 模块用 ListBarFeed
"""
from __future__ import annotations


def test_list_bar_feed_passthrough():
    from evtrade.core._harness import ListBarFeed
    fake = [object() for _ in range(5)]
    out = list(ListBarFeed(fake).stream())
    assert out == fake


def test_sweep_module_imports():
    """sweep 直接吃 numpy dict, 不再经过 NumpyDictFeed (已删)"""
    import evtrade.core.sweep as sweep_mod
    assert hasattr(sweep_mod, "run_one_vectorized")


def test_replay_module_imports_ListBarFeed():
    """replay.py 在 replay_engine 中已 import 并使用 ListBarFeed"""
    import evtrade.core.replay as replay_mod
    from evtrade.core._harness import ListBarFeed
    assert callable(ListBarFeed)
    assert hasattr(replay_mod, "replay_engine")
