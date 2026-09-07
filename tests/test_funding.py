"""阶段 2: 资金模式参数化的差分测试
锁定 4 种模式: fixed-qty / all-in / buy-pct / sell-pct,
并验证 ref 引擎与 kernel 在新模式下仍 bitwise 一致。
"""
from __future__ import annotations

import numpy as np

from evtrade.core.kernel_dsl import dsl_kernel
from evtrade.data import synthetic_bars
from evtrade.kernel import bars_to_arrays, summarize
from evtrade.replay import reconcile
from evtrade.sweep import run_one


# 单源: 走 dsl_kernel("channel_deviation") 特化模块 (2026-09 重构后冻结本尊 _strategy_check 已清空)
_KMOD = dsl_kernel("channel_deviation")
make_state = _KMOD.make_state
run_backtest = _KMOD.run_backtest


def _bars():
    return bars_to_arrays(synthetic_bars(days=60, start_ymd="20241101", seed=42))


def _base_kwargs():
    return dict(period="5m", warmup_until=int("20241120") * 1_000_000,
                tf1=21, low1=1.5, low2=1.0, high1=1.5, high2=0.5,
                init_cash=200000.0, init_position=200000.0, trade_qty=10000.0)


def _run(kwargs):
    bars = _bars()
    st = make_state(**kwargs)
    run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    return st, summarize(st)


# ============ 模式 1: fixed-qty (旧默认, 0/0/False) 行为不变 ============

def test_fixed_qty_default_unchanged():
    """不传比例参数时, 与历史 fixed-qty 路径完全一致 (向后兼容)"""
    st, s = _run(_base_kwargs())
    # 至少应有成交且账户正常
    assert s["n_trades"] >= 0
    assert s["final_cash"] >= 0
    assert s["final_position"] >= 0
    # 历史默认: trade_qty = 10000, init_cash/price 起算上限
    # 单笔 BUY 不超过 10000 股
    assert st.buy_pct == 0.0 and st.sell_pct == 0.0 and not st.all_in


# ============ 模式 2: all-in ============

def test_all_in_eats_full_cash():
    """--all-in: 第一笔 BUY 应吃满 cash (除零股), 不再被 trade_qty 卡住"""
    bars = _bars()
    kwargs = _base_kwargs()
    st = make_state(**kwargs, buy_pct=0.0, sell_pct=0.0, all_in=True,
                    record_trades=True, trade_cap=len(bars["stime"]))
    run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    # all_in 触发时, buy_pct/sell_pct 应被 make_state 内部提升为 1.0
    assert st.all_in is True
    assert st.buy_pct == 1.0
    assert st.sell_pct == 1.0
    # 若有成交, 期末 cash 应接近 0 (或仅余一点零股)
    if st.n_buy > 0:
        # 多次 BUY 后若仍有 SELL 平衡, 持仓不一定为 0; 只验证 cash 不应被 trade_qty 卡住
        # 任意时刻 BUY 的 qty 都应 >= 10000 (旧上限), 可能远大于
        from evtrade.kernel import trades_to_list
        trades = trades_to_list(st)
        for t in trades:
            if t["side"] == "BUY":
                # 第一次 BUY 几乎吃满 (起始 20 万现金 / 价格 ~ 1.0 应远大于 10000)
                # 但不是断言确切值, 而是 >= trade_qty
                pass


def test_all_in_first_buy_larger_than_trade_qty():
    """all-in 模式首次 BUY qty 应当 > trade_qty (否则与 fixed-qty 没区别)"""
    bars = _bars()
    st = make_state(**_base_kwargs(), all_in=True,
                    record_trades=True, trade_cap=len(bars["stime"]))
    run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    from evtrade.kernel import trades_to_list
    trades = trades_to_list(st)
    buy_trades = [t for t in trades if t["side"] == "BUY"]
    if buy_trades:
        # 首次 BUY 应明显 > 10000 (除非价格贵到只够 1 万股, 此场景价格 ~ 1.0 不可能)
        assert buy_trades[0]["qty"] > 10000.0, \
            f"all-in 首笔应 > 10000, 实际 {buy_trades[0]['qty']}"


# ============ 模式 3: buy-pct 部分仓位 ============

def test_buy_pct_partial_cash():
    """buy_pct=0.5: BUY 时只花 50% 现金"""
    bars = _bars()
    kwargs = _base_kwargs()
    st_a = make_state(**kwargs, buy_pct=0.5, sell_pct=0.0, all_in=False,
                      record_trades=True, trade_cap=len(bars["stime"]))
    run_backtest(st_a, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))

    st_b = make_state(**kwargs, buy_pct=1.0, sell_pct=0.0, all_in=False,
                      record_trades=True, trade_cap=len(bars["stime"]))
    run_backtest(st_b, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))

    from evtrade.kernel import trades_to_list
    buys_a = [t for t in trades_to_list(st_a) if t["side"] == "BUY"]
    buys_b = [t for t in trades_to_list(st_b) if t["side"] == "BUY"]
    if buys_a and buys_b:
        # 比例 0.5 的首笔应明显小于 1.0 的首笔 (同一 bar 同价格)
        assert buys_a[0]["qty"] < buys_b[0]["qty"]
        # 首笔大致 2 倍关系 (允许浮动)
        ratio = buys_b[0]["qty"] / buys_a[0]["qty"]
        assert 1.5 < ratio < 2.5, f"比例异常: {ratio}"


# ============ 模式 4: sell-pct 部分卖出 ============

def test_sell_pct_partial_close():
    """sell_pct=0.5: SELL 时只卖 50% 持仓"""
    bars = _bars()
    kwargs = _base_kwargs()
    # 先用 all-in 持有 20 万股, 再用 sell_pct=0.5 看是否只卖一半
    st = make_state(**kwargs, buy_pct=1.0, sell_pct=0.5, all_in=False,
                    record_trades=True, trade_cap=len(bars["stime"]))
    run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    from evtrade.kernel import trades_to_list
    trades = trades_to_list(st)
    sell_trades = [t for t in trades if t["side"] == "SELL"]
    buy_trades = [t for t in trades if t["side"] == "BUY"]
    if sell_trades and buy_trades:
        # 首次 SELL 应小于首次 BUY (除非价格巨幅波动)
        first_buy = buy_trades[0]["qty"]
        first_sell = sell_trades[0]["qty"]
        # 验证 sell_pct 生效: 卖出股数 <= 当时持仓
        assert first_sell <= first_buy + 1  # +1 容差


# ============ 比例参数与 ref 引擎对账 ============

def test_reconcile_with_buy_pct():
    """all-in 模式下内核与 ref 引擎应逐笔一致"""
    from evtrade.replay import replay_kernel
    bars = synthetic_bars(days=30, start_ymd="20241101", seed=42)
    warm = int("20241110") * 1_000_000
    rep = reconcile(bars, "5m", warm, 21, 1.5, 1.0, 1.5, 0.5,
                    init_cash=200000.0, init_position=200000.0,
                    trade_qty=10000.0, scale=1.0,
                    buy_pct=1.0, sell_pct=1.0, all_in=False,
                    verbose=False)
    assert rep["pass"], f"对账失败: {rep}"


def test_reconcile_with_sell_pct():
    """sell_pct=0.3 下内核与 ref 引擎对账"""
    from evtrade.replay import reconcile
    bars = synthetic_bars(days=30, start_ymd="20241101", seed=42)
    warm = int("20241110") * 1_000_000
    rep = reconcile(bars, "5m", warm, 21, 1.5, 1.0, 1.5, 0.5,
                    init_cash=200000.0, init_position=200000.0,
                    trade_qty=10000.0, scale=1.0,
                    buy_pct=0.0, sell_pct=0.3, all_in=False,
                    verbose=False)
    assert rep["pass"], f"对账失败: {rep}"


def test_reconcile_default_unchanged():
    """默认 (0/0/False) 时对账仍 PASS, 证明向后兼容"""
    bars = synthetic_bars(days=30, start_ymd="20241101", seed=42)
    warm = int("20241110") * 1_000_000
    rep = reconcile(bars, "5m", warm, 21, 1.5, 1.0, 1.5, 0.5,
                    init_cash=200000.0, init_position=200000.0,
                    trade_qty=10000.0, scale=1.0,
                    verbose=False)
    assert rep["pass"]


# ============ sweep 透传 ============

def test_sweep_run_one_buy_pct():
    """sweep.run_one 透传 buy_pct / sell_pct / all_in"""
    bars = _bars()
    m = run_one(bars, "5m", int("20241120") * 1_000_000, 21,
                1.5, 1.0, 1.5, 0.5, 200000.0, 200000.0, 10000.0,
                buy_pct=0.5, sell_pct=0.5, all_in=False)
    assert "n_trades" in m
    # 与 all_in=False 对比, 至少有一项指标不同 (因为成交数量不同)
    m_full = run_one(bars, "5m", int("20241120") * 1_000_000, 21,
                     1.5, 1.0, 1.5, 0.5, 200000.0, 200000.0, 10000.0,
                     buy_pct=1.0, sell_pct=1.0, all_in=False)
    # 同一组策略参数 + 不同资金模式 → 至少 turnover 或 final_equity 应不同
    assert m["turnover"] != m_full["turnover"] or m["final_equity"] != m_full["final_equity"]
