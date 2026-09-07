from __future__ import annotations
"""内核单元测试: 空数据 / 容量护栏 / 实盘单步与批量一致性 / 1d 语义"""

import numpy as np

from evtrade.core.kernel_dsl import dsl_kernel
from evtrade.data import synthetic_bars
from evtrade.kernel import (bars_to_arrays, summarize, trades_to_list)


# 单源: 走 dsl_kernel("channel_deviation") 特化模块 (2026-09 重构后冻结本尊 _strategy_check 已清空)
_KMOD = dsl_kernel("channel_deviation")
make_state = _KMOD.make_state
run_backtest = _KMOD.run_backtest
step = _KMOD.step


def test_empty_data_safe():
    """空数组: 安全返回零交易 (参考实现在此场景会 IndexError, 内核为已知改进)"""
    st = make_state(record_trades=True, trade_cap=10)
    empty = {k: np.empty(0, np.int64 if k == "stime" else np.float64)
             for k in ("stime", "open", "high", "low", "close", "volume")}
    run_backtest(st, empty["stime"], empty["open"], empty["high"], empty["low"],
                 empty["close"], empty["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    s = summarize(st)
    assert s["n_trades"] == 0
    assert s["final_equity"] == 200000.0          # last_price=0 -> 持仓市值 0
    assert s["baseline"] == 200000.0
    assert s["excess_pct"] == 0


def test_trade_capacity_guard():
    """成交记录容量不足: 计数完整, 只落前 cap 笔明细"""
    bars = synthetic_bars(days=20, start_ymd="20250101", seed=11)
    arr = bars_to_arrays(bars)
    st = make_state(period="5m", warmup_until=20250106000000, tf1=5,
                    p0=0.3, p1=0.2, p2=0.3, p3=0.15,
                    record_trades=True, trade_cap=1)
    run_backtest(st, arr["stime"], arr["open"], arr["high"], arr["low"],
                 arr["close"], arr["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    assert st.n_trades >= 2
    assert len(trades_to_list(st)) == 1           # 只保留第 1 笔明细


def test_live_step_equals_batch():
    """实盘单步 step() 与批量 run_backtest() 终态一致 (同一代码路径)"""
    bars = synthetic_bars(days=15, start_ymd="20250101", seed=19)
    arr = bars_to_arrays(bars)
    n = len(bars)

    st_batch = make_state(period="5m", warmup_until=20250106000000, tf1=21,
                          p0=0.4, p1=0.25, p2=0.4, p3=0.2)
    run_backtest(st_batch, arr["stime"], arr["open"], arr["high"], arr["low"],
                 arr["close"], arr["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))

    st_live = make_state(period="5m", warmup_until=20250106000000, tf1=21,
                         p0=0.4, p1=0.25, p2=0.4, p3=0.2)
    signals = []
    for i in range(n):
        sig, up, dw = step(st_live, int(arr["stime"][i]), float(arr["open"][i]),
                           float(arr["high"][i]), float(arr["low"][i]),
                           float(arr["close"][i]), float(arr["volume"][i]))
        signals.append(sig)
    assert st_live.cash == st_batch.cash
    assert st_live.position == st_batch.position
    assert st_live.last_price == st_batch.last_price
    assert st_live.n_trades == st_batch.n_trades
    assert max(map(abs, signals)) <= 1


def test_scale_martingale_sequence():
    """倍投系数: 连续同向信号数量累乘 (×scale), 反向重置; 与手工模型逐笔一致

    直接用 sig_out 轨迹回放 kernel._execute 的规格 (含资金/持仓上限),
    校验成交流水的数量、价格、cash_after 逐笔相等。
    """
    bars = synthetic_bars(days=25, start_ymd="20250101", seed=11)
    arr = bars_to_arrays(bars)
    n = len(bars)
    trade_qty, scale = 1000.0, 2.0
    st = make_state(period="5m", warmup_until=20250106000000, tf1=5,
                    p0=0.3, p1=0.2, p2=0.3, p3=0.15,
                    init_cash=200000.0, init_position=200000.0,
                    trade_qty=trade_qty, scale=scale,
                    record_trades=True, trade_cap=n)
    sig_out = np.zeros(n, np.int8)
    run_backtest(st, arr["stime"], arr["open"], arr["high"], arr["low"],
                 arr["close"], arr["volume"], sig_out, np.empty(0), np.empty(0))
    trades = trades_to_list(st)

    # Python 模型: 与 kernel._execute 同式 (成交 ts 记桶右端点, 与内核一致)
    from evtrade.kernel import bucket_ts_encoded, resolve_period_seconds
    P = resolve_period_seconds("5m")
    cash, pos = 200000.0, 200000.0
    last_side, cur_qty = 0, trade_qty
    model = []
    for i in range(n):
        sig = int(sig_out[i])
        if sig == 0:
            continue
        price, ts = float(arr["close"][i]), int(bucket_ts_encoded(int(arr["stime"][i]), P))
        if sig == last_side:
            cur_qty = cur_qty * scale
        else:
            cur_qty = trade_qty
            last_side = sig
        if sig == 1:
            q = min(cur_qty, cash / price) if price > 0 else 0.0
            if q <= 0:
                continue
            cash -= q * price
            pos += q
            side = "BUY"
        else:
            q = min(cur_qty, pos)
            if q <= 0:
                continue
            cash += q * price
            pos -= q
            side = "SELL"
        model.append((ts, side, q, price, cash))

    assert len(trades) == len(model) > 0
    for t, m in zip(trades, model):
        assert t["ts"] == m[0] and t["side"] == m[1]
        assert t["qty"] == m[2], f"qty: model={m[2]!r} kernel={t['qty']!r}"
        assert t["price"] == m[3]
        assert t["cash_after"] == m[4]
    assert any(m[2] > trade_qty for m in model), "倍投未生效"


def test_bucket_table():
    """桶表 (framework 层): OHLCV 聚合 / 通道轨 / 信号计数

    策略专属偏离指标 (channel_deviation 的 low_dev / high_dev / ...) 不在框架中,
    改由 strategies/channel_deviation.py::compute_deviation_columns 提供,
    这里只校验框架桶聚合本身。
    """
    from evtrade.kernel import bucket_table, run_backtest_trace
    bars = synthetic_bars(days=10, start_ymd="20250101", seed=5)
    arr = bars_to_arrays(bars)
    n = len(bars)
    st = make_state(period="5m", warmup_until=0, tf1=3,
                    p0=0.5, p1=0.3, p2=0.5, p3=0.3,
                    trade_qty=1000.0, scale=2.0)
    sig = np.zeros(n, np.int8)
    up = np.full(n, np.nan)
    dw = np.full(n, np.nan)
    ts_out = np.zeros(n, np.int64)
    o_out = np.zeros(n)
    h_out = np.zeros(n)
    l_out = np.zeros(n)
    c_out = np.zeros(n)
    v_out = np.zeros(n)
    run_backtest_trace(st, arr["stime"], arr["open"], arr["high"], arr["low"],
                       arr["close"], arr["volume"], sig, up, dw,
                       ts_out, o_out, h_out, l_out, c_out, v_out)
    tab = bucket_table(arr["stime"], sig, up, dw,
                       ts_out, o_out, h_out, l_out, c_out, v_out)

    # 框架只输出: OHLCV + count + up/dw + sig + n_sig
    expected_keys = {"ts", "open", "high", "low", "close", "volume", "count",
                     "up", "dw", "sig", "n_sig"}
    assert set(tab.keys()) == expected_keys, \
        f"bucket_table 应只输出 framework 字段, 多了: {set(tab.keys()) - expected_keys}"

    uniq_ts, first_idx, counts = np.unique(ts_out, return_index=True,
                                           return_counts=True)
    assert np.array_equal(tab["ts"], uniq_ts)
    assert np.array_equal(tab["count"], counts)
    # 桶 open = 桶首根 bar 的 open; close = 桶末根 bar 的 close; high/low 极值
    assert np.allclose(tab["open"], arr["open"][first_idx])
    last_idx = first_idx + counts - 1
    assert np.allclose(tab["close"], arr["close"][last_idx])
    assert np.allclose(tab["high"], np.maximum.reduceat(arr["high"], first_idx))
    assert np.allclose(tab["low"], np.minimum.reduceat(arr["low"], first_idx))
    assert np.allclose(tab["volume"],
                       np.add.reduceat(arr["volume"], first_idx))
    # 信号计数守恒
    assert int(tab["n_sig"].sum()) == int((sig != 0).sum())
    # 通道轨在 up/dw 无效处 (未就绪) 为 NaN (frame 层面提供原值过滤)
    invalid = ~(np.isfinite(tab["up"]) & np.isfinite(tab["dw"]) & (tab["up"] != 0))
    assert np.isnan(tab["up"][invalid]).all()
    assert np.isnan(tab["dw"][invalid]).all()


def test_period_1d_bucket_example():
    """1d 桶 ts 语义抽查 (前开后闭 -> 次日零点)"""
    from evtrade.kernel import bucket_ts_encoded, resolve_period_seconds
    p = resolve_period_seconds("1d")
    assert int(bucket_ts_encoded(20250105093500, p)) == 20250106000000


def test_excess_curve_metrics():
    """超额曲线指标定义校验: 手工构造 x 序列, 与内核同式逐项重算

    x_t = (策略权益 - 基线权益)/基线权益; 通过直接改账户字段模拟一笔买入,
    让 x 在 4 个交易日形成 波峰->回撤->回升->新高 的形态。
    """
    # step 已通过模块顶部 _KMOD = dsl_kernel("channel_deviation") 引入
    st = make_state(period="5m", warmup_until=0, tf1=5,
                    p0=99.0, p1=98.0, p2=99.0, p3=98.0)

    def feed(ts, c):
        step(st, ts, c, c, c, c, 1000.0)

    feed(20250106093000, 10.0)
    feed(20250106093100, 10.0)
    st.cash, st.position = 100000.0, 210000.0      # 模拟买入后的账户状态
    feed(20250106093200, 11.0)
    feed(20250107093000, 10.9)
    feed(20250107093100, 10.9)
    feed(20250108093000, 11.0)
    feed(20250108093100, 11.0)
    feed(20250109093000, 11.2)
    feed(20250109093100, 11.2)
    feed(20250110093000, 11.2)
    feed(20250110093100, 11.2)

    # 与内核同式重算
    def x_of(p):
        eq = 100000.0 + 210000.0 * p
        base = 200000.0 + 200000.0 * p
        return (eq - base) / base

    peak = x_of(11.0)
    expected_xmdd = peak - x_of(10.9)
    d1 = x_of(10.9) - x_of(11.0)
    d2 = x_of(11.0) - x_of(10.9)
    d3 = x_of(11.2) - x_of(11.0)
    dsum = d1 + d2 + d3
    dsum2 = d1 * d1 + d2 * d2 + d3 * d3
    mean = dsum / 3
    var = dsum2 / 3 - mean * mean
    expected_sharpe = mean / var ** 0.5 * 252.0 ** 0.5

    s = summarize(st)
    assert s["x_mdd"] == expected_xmdd
    assert s["sharpe_excess"] == expected_sharpe
    assert s["sharpe_excess"] != 0.0
    assert 0 < s["years"] < 1
    assert abs(s["ann_excess_pct"] - s["excess_pct"] / s["years"]) < 1e-12
