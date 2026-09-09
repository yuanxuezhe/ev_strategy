from __future__ import annotations
"""录制/回放/对账工具测试"""

import numpy as np
import pytest

from evtrade.data import synthetic_bars
from evtrade.kernel import resolve_period_seconds
from evtrade.replay import (append_bar, diff_signals, read_bars_log,
                            reconcile, replay_engine, replay_kernel,
                            write_bars_log)


def test_log_roundtrip(tmp_path):
    bars = synthetic_bars(days=5, start_ymd="20250101", seed=1)
    path = str(tmp_path / "bars.log")
    n = write_bars_log(path, bars)
    assert n == len(bars)
    back = read_bars_log(path)
    assert len(back) == len(bars)
    for a, b in zip(bars, back):
        assert a.stime == b.stime and a.code == b.code
        assert (a.open, a.high, a.low, a.close, a.volume) == \
               (b.open, b.high, b.low, b.close, b.volume)


def test_append_bar_creates_header_and_appends(tmp_path):
    bars = synthetic_bars(days=2, start_ymd="20250101", seed=2)
    path = str(tmp_path / "live.log")
    append_bar(path, bars[0])
    append_bar(path, bars[1])
    back = read_bars_log(path)
    assert [b.stime for b in back] == [bars[0].stime, bars[1].stime]


def test_read_rejects_unsorted(tmp_path):
    bars = synthetic_bars(days=3, start_ymd="20250101", seed=3)
    path = str(tmp_path / "bad.log")
    write_bars_log(path, bars)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{bars[0].stime},{bars[0].code},1,1,1,1,1\n")  # 重复/乱序
    with pytest.raises(ValueError):
        read_bars_log(path)


def test_replay_kernel_vs_engine_signals_equal():
    """两条引擎路径在同一录制数据上: 逐 bar 信号与通道值一致 (策略期对齐)"""
    bars = synthetic_bars(days=20, start_ymd="20250101", seed=11)
    stime = np.array([int(b.stime) for b in bars])
    warm = 20250106000000
    offset = int(np.searchsorted(stime, warm))
    # tf1 已下沉为策略参数 (channel_deviation 的 params_spec 自声明);
    # 若 strategy_params 不含 tf1, 引擎和内核各自的 tf1 入参会落到不同字段
    # (内核 make_state_general 兜底, 引擎 _sync_strategy_state 不动 ctx.params),
    # 导致三端 EMA 周期不一致, 信号漂移。这里显式填入以锁定两边行为。
    params = {"low1": 0.4, "low2": 0.25, "high1": 0.4, "high2": 0.2, "tf1": 5}
    k = replay_kernel(bars, "5m", warm, 5,
                      strategy_name="channel_deviation", strategy_params=params)
    r = replay_engine(bars, "5m", warm, 5,
                      strategy_name="channel_deviation", strategy_params=params)
    d = diff_signals(k["sig"], r["sig"])
    assert d["n_diff"] == 0, d
    # 内核在预热期也输出通道值; 参考引擎只记录策略期 -> 策略期区间内逐元素比较
    assert np.array_equal(k["per_bar"]["up"][offset:],
                          r["per_bar"]["up"][offset:], equal_nan=True)
    assert np.array_equal(k["per_bar"]["dw"][offset:],
                          r["per_bar"]["dw"][offset:], equal_nan=True)
    assert len(k["trades"]) == len(r["trades"]) > 0


def test_reconcile_pass():
    bars = synthetic_bars(days=15, start_ymd="20250101", seed=19)
    report = reconcile(bars, "5m", 20250105000000, 21,
                       strategy_name="channel_deviation",
                       strategy_params={"low1": 0.4, "low2": 0.25,
                                        "high1": 0.4, "high2": 0.2},
                       verbose=False)
    assert report["pass"] is True
    assert report["sig"]["n_a"] == report["sig"]["n_b"]
    assert report["n_bars"] == len(bars)


def test_diff_signals_counts():
    a = np.array([0, 1, 0, -1, 0])
    b = np.array([0, 1, 1, -1, 0])
    d = diff_signals(a, b)
    assert d["n_diff"] == 1 and d["first_idx"] == 2
    assert diff_signals(np.zeros(3), np.zeros(2))["n_diff"] == -1


def test_replay_period_validation():
    bars = synthetic_bars(days=5, start_ymd="20250101", seed=5)
    with pytest.raises(ValueError):
        replay_kernel(bars, "5x", 0, 21,
                      strategy_name="channel_deviation",
                      strategy_params={"low1": 1.5, "low2": 1.0,
                                       "high1": 1.5, "high2": 0.5})
    assert resolve_period_seconds("5m") == 300
