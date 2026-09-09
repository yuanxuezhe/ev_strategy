"""统一策略契约锁定测试 (DSL/numba 已下线)

核心断言:
  1. channel_deviation / ma_crossover 都是 VectorizedStrategy 子类
  2. compute_signals 返回 xp.ndarray[int8]
  3. cpu vs gpu 信号 bitwise 一致 (cupy 不可用时跳过 gpu)
  4. vectorized 路径 vs Engine.on_bars 路径逐笔一致 (reconcile)
  5. metrics summary 16 字段齐全 (cagr/sharpe/sortino/calmar/max_dd_days/...)
"""
from __future__ import annotations

import numpy as np
import pytest

import evtrade
from evtrade import (
    VectorizedStrategy, get_strategy, run_vectorized,
)
from evtrade.backends import get_xp


def _make_bars(n: int = 200, seed: int = 42):
    """合成 1m bars (np.float64), 24h 内连续"""
    import datetime
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.0005, n)
    prices = 100 * np.exp(np.cumsum(rets))
    high = prices * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = prices * (1 - np.abs(rng.normal(0, 0.001, n)))
    start = datetime.datetime(2025, 1, 1, 9, 30, 0)
    stime = []
    cur = start
    for _ in range(n):
        stime.append(int(cur.strftime("%Y%m%d%H%M%S")))
        cur = cur + datetime.timedelta(minutes=1)
    return {
        "stime": np.array(stime, dtype=np.int64),
        "open": prices.astype(np.float64),
        "high": high.astype(np.float64),
        "low": low.astype(np.float64),
        "close": prices.astype(np.float64),
        "volume": np.ones(n, dtype=np.float64),
    }


def test_both_strategies_are_vectorized_subclass():
    assert issubclass(evtrade.ChannelDeviationStrategy, VectorizedStrategy)
    assert issubclass(evtrade.MACrossoverStrategy, VectorizedStrategy)


def test_strategy_registry_lists_both():
    keys = evtrade.available_strategies()
    assert "channel_deviation" in keys
    assert "ma_crossover" in keys


def test_compute_signals_returns_int8_xp_array():
    """compute_signals(xp, bars, params) -> xp.ndarray[int8]"""
    s = evtrade.get_strategy("ma_crossover", fast=3, slow=10)
    bars = _aggregate_bars("5m", _make_bars())
    sig = s.compute_signals(np, bars, s.params)
    assert sig.dtype == np.int8
    assert len(sig) == len(bars["ts"])


def test_ma_crossover_cpu_vs_gpu_bitwise_equal():
    """CPU vs GPU 信号 bitwise 一致 (cupy 不可用时 skip)"""
    try:
        cp = get_xp("gpu")
        _ = cp.zeros(2)
    except Exception:
        pytest.skip("cupy/CUDA 不可用")

    s_cpu = evtrade.get_strategy("ma_crossover", fast=5, slow=20)
    s_gpu = evtrade.get_strategy("ma_crossover", fast=5, slow=20)
    bars_cpu = _aggregate_bars("5m", _make_bars())
    bars_gpu = _aggregate_bars_gpu("5m", _make_bars())

    sig_cpu = s_cpu.compute_signals(np, bars_cpu, s_cpu.params)
    sig_gpu = s_gpu.compute_signals(cp, bars_gpu, s_gpu.params).get()
    np.testing.assert_array_equal(sig_cpu, sig_gpu)


def test_vectorized_vs_engine_on_bars_reconcile():
    """vectorized 路径 vs Engine.on_bars 路径逐笔一致 (channel_deviation)"""
    from evtrade.account import Account
    from evtrade.aggregator import BarAggregator
    from evtrade.core._harness import ListBarFeed
    from evtrade.core.engine import Engine
    from evtrade.core.replay import reconcile
    from evtrade.core.timeutils import resolve_period_seconds
    from evtrade.execution import SimulatedExecutor
    from evtrade.primitives import Bar

    bars_arr = _make_bars(n=300, seed=7)
    n_bars = len(bars_arr["stime"])
    # vectorized 路径 (桶级信号)
    strat = evtrade.get_strategy("channel_deviation", tf1=5)
    out_v = run_vectorized(bars_arr, "5m", warmup_until=0,
                           strategy=strat, params=strat.params, device="cpu")
    sig_v = out_v["sig"]      # 桶级
    trades_v = out_v["summary"]["trades"]

    # Engine.on_bars 路径 (1m bar -> aggregator 内部桶 -> on_bars 调用 = 桶级)
    bars_list = [Bar(stime=str(bars_arr["stime"][i]), code="T",
                     open=float(bars_arr["open"][i]),
                     high=float(bars_arr["high"][i]),
                     low=float(bars_arr["low"][i]),
                     close=float(bars_arr["close"][i]),
                     volume=int(bars_arr["volume"][i]))
                 for i in range(n_bars)]
    # 重建 strategy (避免 instance FSM 串味)
    strat2 = evtrade.get_strategy("channel_deviation", tf1=5)
    feed = ListBarFeed(bars_list)
    agg = BarAggregator(resolve_period_seconds("5m"), on_bars=None)
    exe = SimulatedExecutor(Account(cash=100000, position=0), qty=100, verbose=False)
    eng = Engine(feed, agg, strat2, exe, verbose=False)
    eng.run()
    sig_r = np.array(eng.bucket_signals, dtype=np.int8)  # 桶级

    # vectorized vs Engine 桶级信号: 偶发 1 根边界差异 (warmup_until=0 时
    # bucket_ts 与 timeutils 算的 ts 在第一根有 1 桶差, 见 kbs/13 §5);
    # 严口径锁 0, 宽口径锁 <=1 (默认宽口径; 严口径用 env EVT_RECONCILE_STRICT=1)
    import os
    n = min(len(sig_v), len(sig_r))
    n_diff = int(np.sum(sig_v[:n] != sig_r[:n]))
    cap = 0 if os.environ.get("EVT_RECONCILE_STRICT") == "1" else 1
    assert n_diff <= cap, f"桶级信号分歧 {n_diff}/{n} (cap={cap})"

    # 成交笔数大致一致 (允许桶边界 1 根差异: 见 kbs/13 §5)
    trades_r = exe.account.trades
    trade_diff = abs(len(trades_v) - len(trades_r))
    assert trade_diff <= 1, \
        f"成交笔数差异过大 vectorized={len(trades_v)} engine={len(trades_r)}"


def test_metrics_summary_has_16_fields():
    """run_vectorized.summary 必须含 cagr/sharpe_excess/sortino_excess/calmar/
    max_dd_days/max_dd_recovered/x_mdd/max_drawdown 等 16+ 字段"""
    strat = evtrade.get_strategy("ma_crossover", fast=5, slow=20)
    out = run_vectorized(_make_bars(n=300), "5m", warmup_until=0,
                         strategy=strat, params=strat.params, device="cpu")
    s = out["summary"]
    required = {
        "final_price", "final_cash", "final_position", "final_equity",
        "baseline", "excess", "excess_pct", "years",
        "n_trades", "n_buy", "n_sell", "turnover",
        "cagr", "sharpe_excess", "sortino_excess", "calmar",
        "max_dd_days", "max_dd_recovered", "x_mdd", "max_drawdown",
    }
    missing = required - set(s.keys())
    assert not missing, f"summary 缺字段: {sorted(missing)}"
    assert s["years"] > 0 or s["n_trades"] == 0  # 退化场景下 years 可为 0


def test_compute_signals_for_one_bar_default_wrapper():
    """默认包装: 单 bar -> 单元素数组 -> compute_signals[0]"""

    class TestStrat(VectorizedStrategy):
        params_spec = {}

        def compute_signals(self, xp, bars, params):
            return xp.ones(len(bars["ts"]), dtype=xp.int8)

    bar = {"ts": 100, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0,
           "v": 0.0, "mark": 1}
    s = TestStrat()
    assert s.compute_signals_for_one_bar(np, bar, {}) == 1


# ============ helpers ============

def _aggregate_bars(period: str, bars: dict) -> dict:
    """跑一次 vectorized_engine._aggregate_buckets_xp (numpy)"""
    from evtrade.core.vectorized_engine import _aggregate_buckets_xp
    return _aggregate_buckets_xp(np, bars, period, warmup_until=0)


def _aggregate_bars_gpu(period: str, bars: dict) -> dict:
    """GPU 版桶聚合"""
    cp = get_xp("gpu")
    from evtrade.core.vectorized_engine import _aggregate_buckets_xp
    return _aggregate_buckets_xp(cp, bars, period, warmup_until=0)
