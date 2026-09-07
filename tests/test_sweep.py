from __future__ import annotations
"""参数扫描测试: 网格解析 / 并发结果 / walk-forward / 滚动多窗 / 评分框架"""

import numpy as np

from evtrade.data import synthetic_bars
from evtrade.kernel import bars_to_arrays
from evtrade.core.kernel_dsl import dsl_kernel
from evtrade.sweep import (_ann_net, _neighbor_decay, _pareto_flag, parse_grid,
                           run_one, sweep)


# 单源: 走 dsl_kernel("channel_deviation") 特化模块 (2026-09 重构后冻结本尊 _strategy_check 已清空)
_KMOD = dsl_kernel("channel_deviation")
make_state = _KMOD.make_state
run_backtest = _KMOD.run_backtest
trades_to_list = _KMOD.trades_to_list


def _bars():
    return bars_to_arrays(synthetic_bars(days=30, start_ymd="20241201", seed=42))


def _base():
    return {"start": "20241210", "period": "5m", "tf1": 21,
            "low1": 0.4, "low2": 0.25, "high1": 0.4, "high2": 0.2,
            "trade_qty": 10000.0, "init_cash": 200000.0, "init_position": 200000.0}


def test_parse_grid_cartesian():
    combos = parse_grid(["low1=0.4,0.6", "high2=0.2,0.3"])
    assert combos == [
        {"low1": 0.4, "high2": 0.2}, {"low1": 0.4, "high2": 0.3},
        {"low1": 0.6, "high2": 0.2}, {"low1": 0.6, "high2": 0.3},
    ]


def test_parse_grid_tf1_and_period():
    combos = parse_grid(["tf1=10,21", "period=5m,1h"])
    assert combos[0] == {"tf1": 10, "period": "5m"}
    assert combos[-1] == {"tf1": 21, "period": "1h"}


def test_parse_grid_funding_keys():
    """阶段 2: all_in / buy_pct / sell_pct 作为网格维度"""
    combos = parse_grid(["all_in=true,false", "buy_pct=0.3,0.5,1.0"])
    assert len(combos) == 6
    assert combos[0]["all_in"] is True
    assert combos[0]["buy_pct"] == 0.3
    assert combos[1]["all_in"] is True
    assert combos[1]["buy_pct"] == 0.5
    assert combos[3]["all_in"] is False
    assert combos[5] == {"all_in": False, "buy_pct": 1.0}


def test_parse_grid_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        parse_grid(["nope=1,2"])


def test_sweep_small_grid():
    bars = _bars()
    combos = parse_grid(["low1=0.3,0.4", "high1=0.3,0.4"])
    df = sweep(bars, _base(), combos, n_workers=4)
    assert len(df) == 4
    for col in ("n_trades", "final_equity", "baseline", "excess", "excess_pct",
                "years", "ann_excess_pct", "sharpe_excess", "x_mdd",
                "ann_net", "ann_net_min", "ann_net_mean", "pos_ratio",
                "sharpe_min", "S", "score", "pareto", "filter_pass"):
        assert col in df.columns, col
    # 阶段 1 新增列
    for col in ("sortino_excess", "cagr", "calmar", "max_dd_days",
                "max_dd_recovered", "sortino_min", "calmar_max",
                "cagr_max", "max_dd_days_max"):
        assert col in df.columns, col
    assert df["score"].is_monotonic_decreasing
    # 抽一组与直接调用核对 (原始指标 + 年化扣费口径)
    row = df[(df["low1"] == 0.3) & (df["high1"] == 0.3)].iloc[0]
    direct = run_one(bars, "5m", int(_base()["start"]) * 1_000_000, 21,
                     0.3, 0.25, 0.3, 0.2, 200000.0, 200000.0, 10000.0)
    assert row["n_trades"] == direct["n_trades"]
    assert row["excess"] == direct["excess"]
    assert row["ann_net"] == _ann_net(direct, 5.0)
    assert row["score"] == row["ann_net_min"] / (1.0 + 1.0 * row["S"]) \
        if row["ann_net_min"] > 0 else row["score"] == row["ann_net_min"]


def test_sweep_walk_forward():
    bars = _bars()
    base = {**_base(), "start": "20241201"}
    combos = parse_grid(["low1=0.3,0.5"])
    df = sweep(bars, base, combos, split_ymd="20241220", n_workers=2)
    assert len(df) == 2
    assert "train_excess_pct" in df.columns and "test_excess_pct" in df.columns
    assert "test_ann_net" in df.columns and "score" in df.columns
    row = df[df["low1"] == 0.3].iloc[0]
    assert 0 < row["train_n_trades"]
    assert 0 < row["test_n_trades"]
    # 测试窗与全窗口都终于同一根 bar -> 期末价一致
    total = run_one(bars, "5m", int(base["start"]) * 1_000_000, 21,
                    0.3, 0.25, 0.4, 0.2, 200000.0, 200000.0, 10000.0)
    assert row["test_final_price"] == total["final_price"]
    # 注: train_n + test_n 与总窗口成交数可差分界处的锁存跨越成交——
    # 测试窗从 split 起算时锁存状态全新, 属预期 (正是 walk-forward 想要的独立起算)。


def test_sweep_rolling_splits():
    bars = _bars()
    base = {**_base(), "start": "20241201"}
    combos = parse_grid(["low1=0.3,0.5"])
    df = sweep(bars, base, combos, splits=["20241215", "20241222"], n_workers=2)
    assert len(df) == 2
    for col in ("train_excess_pct", "test1_excess_pct", "test2_excess_pct",
                "test1_ann_net", "test2_ann_net", "ann_net_min", "pos_ratio",
                "sharpe_min", "x_mdd_max", "S", "score", "pareto"):
        assert col in df.columns, col
    assert df["score"].is_monotonic_decreasing
    row = df.iloc[0]
    assert row["ann_net_min"] == min(row["test1_ann_net"], row["test2_ann_net"])
    assert 0.0 <= row["pos_ratio"] <= 1.0
    assert 0.0 <= row["S"] <= 1.0


def test_neighbor_decay():
    combos = [{"x": 1}, {"x": 2}, {"x": 3}]
    # 尖峰在 x=3 (邻居都差), 平原在 x=1..2
    S = _neighbor_decay(combos, np.array([1.0, 1.0, 5.0]))
    assert S[0] == 0.0            # 邻居 1.0, 自身 1.0 -> 无衰减
    assert S[1] == 0.0            # 邻居 1.0/5.0, 均值 3.0 > 自身 1.0 -> 截断为 0
    assert 0.0 < S[2] <= 1.0      # 邻居均值 1.0 vs 自身 5.0 -> 高衰减
    assert S[2] == 1.0 - 1.0 / 5.0
    # self<=0 -> 0
    S2 = _neighbor_decay(combos, np.array([-1.0, -1.0, -1.0]))
    assert (S2 == 0).all()


def test_pareto_flag():
    a = np.array([5.0, 4.0, 5.0, 1.0])   # 期望收益
    b = np.array([1.0, 2.0, 2.0, 2.0])   # Sharpe
    flags = _pareto_flag(a, b)
    # (5,2) 支配其他全部; (5,1) 与 (4,2) 被支配; (1,2) 被支配
    assert flags[2] and not flags[0] and not flags[1] and not flags[3]


def test_ann_net_math():
    m = {"excess_pct": 10.0, "years": 2.0, "turnover": 100000.0, "baseline": 400000.0}
    # 10%/2年 = 5%/年; 总费用 100000×5bp = 50 元 = 基线的 0.0125% -> 年化 0.00625%
    assert abs(_ann_net(m, 5.0) - (5.0 - 0.00625)) < 1e-9
    assert _ann_net({"years": 0}, 5.0) == 0.0


def test_permutation_sanity():
    """蒙特卡洛置换检验: p 值在 (0,1], 真实值 (费前年化) 可复现"""
    from evtrade.permutation import permutation_test
    bars = _bars()
    params = {**_base()}
    warm = int(_base()["start"]) * 1_000_000
    r = permutation_test(bars, params, warm, n=30, fee_bp=5.0, seed=7)
    assert 0.0 < r["p_value"] <= 1.0
    assert r["n"] == 30
    assert r["null_p95"] >= r["null_mean"]
    direct = run_one(bars, "5m", warm, 21, 0.4, 0.25, 0.4, 0.2,
                     200000.0, 200000.0, 10000.0)
    assert r["real_ann_net"] == direct["ann_excess_pct"]


def test_test_window_trades_only_after_split():
    """warmup_until=split: 成交必须全部落在测试窗内"""
    bars = _bars()
    split_int = 20241220000000
    st = make_state(period="5m", warmup_until=split_int, tf1=21,
                    low1=0.4, low2=0.25, high1=0.4, high2=0.2,
                    record_trades=True, trade_cap=len(bars["stime"]))
    run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    ts_list = [t["ts"] for t in trades_to_list(st)]
    assert ts_list, "应存在测试窗成交"
    assert all(ts >= split_int for ts in ts_list)
