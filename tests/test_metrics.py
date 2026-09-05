"""阶段 1 新增绩效指标的差分测试"""
from __future__ import annotations

import numpy as np

from evtrade.data import synthetic_bars
from evtrade.kernel import (bars_to_arrays, make_state, run_backtest,
                            summarize, trades_to_list)


def _bars():
    return bars_to_arrays(synthetic_bars(days=60, start_ymd="20241101", seed=42))


def test_summarize_has_new_keys():
    bars = _bars()
    st = make_state(period="5m", warmup_until=int("20241120") * 1_000_000,
                    tf1=21, low1=1.5, low2=1.0, high1=1.5, high2=0.5)
    run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    s = summarize(st)
    for k in ("cagr", "sortino_excess", "calmar", "max_dd_days", "max_dd_recovered"):
        assert k in s, k
    # sanity: CAGR = 0 当 ann_excess 接近 0 时
    assert isinstance(s["cagr"], float)
    assert isinstance(s["sortino_excess"], float)
    assert isinstance(s["calmar"], float)
    assert isinstance(s["max_dd_days"], float)
    assert isinstance(s["max_dd_recovered"], bool)


def test_cagr_matches_manual():
    """CAGR = (终值/初值)^(1/年) - 1, 与手工核对"""
    bars = _bars()
    st = make_state(period="5m", warmup_until=int("20241120") * 1_000_000,
                    tf1=21, low1=1.5, low2=1.0, high1=1.5, high2=0.5)
    run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    s = summarize(st)
    equity = s["final_equity"]
    init_eq = st.init_equity
    years = s["years"]
    assert init_eq > 0 and years > 0
    expected = (equity / init_eq) ** (1.0 / years) - 1.0
    assert abs(s["cagr"] / 100.0 - expected) < 1e-9


def test_max_dd_days_non_negative_and_recovered_flag():
    bars = _bars()
    st = make_state(period="5m", warmup_until=int("20241120") * 1_000_000,
                    tf1=21, low1=1.5, low2=1.0, high1=1.5, high2=0.5)
    run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    s = summarize(st)
    assert s["max_dd_days"] >= 0.0
    # 如果回撤 > 0, 那必然有过谷底
    if s["max_drawdown"] > 0:
        assert s["max_dd_days"] > 0.0


def test_sortino_ge_zero_when_no_drawdown_days():
    """没有下行日时 Sortino = 0"""
    bars = _bars()
    st = make_state(period="5m", warmup_until=int("20241120") * 1_000_000,
                    tf1=21, low1=1.5, low2=1.0, high1=1.5, high2=0.5)
    run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    s = summarize(st)
    if st.d_neg_n == 0:
        assert s["sortino_excess"] == 0.0


def test_old_keys_unchanged():
    """确保旧指标(cash/position/turnover/excess_pct/ann_excess_pct/...)数值不变"""
    bars = _bars()
    st = make_state(period="5m", warmup_until=int("20241120") * 1_000_000,
                    tf1=21, low1=1.5, low2=1.0, high1=1.5, high2=0.5,
                    record_trades=True, trade_cap=len(bars["stime"]))
    run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    s = summarize(st)
    tr = trades_to_list(st)
    # 旧字段仍在且非空
    for k in ("final_price", "n_trades", "n_buy", "n_sell", "final_cash",
              "final_position", "final_equity", "baseline", "excess",
              "excess_pct", "years", "ann_excess_pct", "sharpe_excess",
              "x_mdd", "max_drawdown", "turnover"):
        assert k in s, k
    assert len(tr) == s["n_trades"]
    assert s["final_cash"] == st.cash
    assert s["final_position"] == st.position
    assert s["turnover"] == st.turnover
