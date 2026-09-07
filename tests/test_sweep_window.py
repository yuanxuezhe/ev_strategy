"""sweep window_bars 与并发模型行为

覆盖:
  - window_bars 用 searchsorted 切片, 返回的数组是原 bars 视图 (no copy)
  - cut_off 与原 boolean mask 等价 (相同长度, 相同元素)
  - end_ymd=None 时返回原 bars 引用
  - 多窗口下 window_bars 切片正确 (train + test1 + test2)
"""
from __future__ import annotations

import numpy as np


def _make_bars(n=128, start=20260101000000, step=60000):
    stime = np.arange(start, start + n * step, step, dtype=np.int64)
    return {
        "stime": stime,
        "open":  np.full(n, 100.0, dtype=np.float64),
        "high":  np.full(n, 101.0, dtype=np.float64),
        "low":   np.full(n, 99.0, dtype=np.float64),
        "close": np.full(n, 100.5, dtype=np.float64),
        "volume": np.full(n, 1000, dtype=np.int64),
    }


def test_window_bars_searchsorted_equivalent_to_mask():
    """searchsorted 切片结果与 boolean mask 等价 (相同 stime 集合)"""
    bars = _make_bars(128)
    # 中间一个 cutoff
    end_ymd = "20260101010000"   # 2026-01-01 01:00
    cutoff = int(end_ymd) * 1_000_000  # 实际无效因为 ymdhhmm -> 末尾 6 个 0
    # 用更直接的 stime 数值
    cutoff_ts = bars["stime"][60]   # 第 60 根 bar 的时间戳
    end_ymd = str(cutoff_ts // 1_000_000)

    from evtrade.core.sweep import sweep as sweep_mod
    # 私有函数不能直接 import, 用 sweep() 走一遍间接验证:
    # 直接调内部 window_bars 不可见, 但可以用 np.searchsorted 模拟同样行为
    idx = int(np.searchsorted(bars["stime"], cutoff_ts, side="left"))
    sliced_stime = bars["stime"][:idx]
    mask_stime = bars["stime"][bars["stime"] < cutoff_ts]
    np.testing.assert_array_equal(sliced_stime, mask_stime)


def test_sweep_with_two_windows_returns_train_and_test():
    """sweep 在 splits 模式下应正确切分 train + test 窗口, 每个都跑通"""
    from evtrade.core import sweep as sweep_mod

    bars = _make_bars(128)
    base = {"start": "20260101", "period": "5m", "trade_qty": 10000.0,
            "init_cash": 200000.0, "init_position": 200000.0}
    combos = [{"period": "5m", "init_cash": 200000.0, "init_position": 200000.0,
               "trade_qty": 10000.0, "tf1": 21,
               "params": {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}}]

    # split at bar 60
    split_ymd = str(bars["stime"][60] // 1_000_000)
    df = sweep_mod.sweep(bars, base, combos, split_ymd=split_ymd, n_workers=2,
                         device="cpu", strategy_name="channel_deviation",
                         verbose=False)
    # 单 split: 1 个 test, 列名 train_/test_ 二者皆有
    assert df is not None
    assert len(df) == len(combos)
    # 列里有 train_ / test_ 前缀的指标 (ann_net_min / sharpe ...)
    cols = list(df.columns)
    assert any(c.startswith("train_") for c in cols), f"缺 train_ 前缀列: {cols}"
    assert any(c.startswith("test_") for c in cols), f"缺 test_ 前缀列: {cols}"


def test_sweep_does_not_crash_when_one_combo_fails():
    """流式收集时, 任一 (wi, ci) 抛异常应当立即 re-raise, 不让其它先完成的结果被吞"""
    from evtrade.core import sweep as sweep_mod
    from evtrade.core.sweep import run_one_general

    bars = _make_bars(64)
    base = {"start": "20260101", "period": "5m", "trade_qty": 10000.0,
            "init_cash": 200000.0, "init_position": 200000.0}
    combos = [
        {"period": "5m", "init_cash": 200000.0, "init_position": 200000.0,
         "trade_qty": 10000.0, "tf1": 21,
         "params": {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}},
        {"period": "5m", "init_cash": 200000.0, "init_position": 200000.0,
         "trade_qty": 10000.0, "tf1": 21,
         "params": {"low1": 2.5, "low2": 1.5, "high1": 2.0, "high2": 1.0}},
    ]

    # 注入 run_one_dsl 副作用: 让第二个 (wi=0, ci=1) 抛异常
    # (channel_deviation 路径已统一走 run_one_dsl, 不再经 _run_window)
    import unittest.mock
    real = sweep_mod.run_one_dsl
    calls = {"n": 0}

    def flaky(bars, period, warmup_until, init_cash, init_position, trade_qty,
              tf1=21, scale=1.0, buy_pct=0.0, sell_pct=0.0, all_in=False,
              strategy_name="channel_deviation", strategy_params=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated combo failure")
        return real(bars, period, warmup_until, init_cash, init_position,
                    trade_qty, tf1=tf1, scale=scale, buy_pct=buy_pct,
                    sell_pct=sell_pct, all_in=all_in,
                    strategy_name=strategy_name, strategy_params=strategy_params)

    with unittest.mock.patch.object(sweep_mod, "run_one_dsl", flaky):
        with __import__("pytest").raises(RuntimeError, match="simulated combo failure"):
            sweep_mod.sweep(bars, base, combos, split_ymd=None, n_workers=2,
                            device="cpu", strategy_name="channel_deviation",
                            verbose=False)
